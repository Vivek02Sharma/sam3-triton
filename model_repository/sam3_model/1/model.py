import triton_python_backend_utils as pb_utils 
from transformers import AutoTokenizer
import cv2
import numpy as np
import asyncio
import time

class TritonPythonModel:
    def initialize(self, args):
        self.logger = pb_utils.Logger
        self.tokenizer = AutoTokenizer.from_pretrained(args["model_repository"] + "/1/tokenizer_assets")
        # sam3 official normalization values
        self.pixel_mean = np.array([0.5, 0.5, 0.5]).reshape(1, 1, 3)
        self.pixel_std = np.array([0.5, 0.5, 0.5]).reshape(1 , 1, 3)
        self.target_size = 1008
        self.cpu_mem = pb_utils.PreferredMemory(pb_utils.TRITONSERVER_MEMORY_CPU)
        self.logger.log("sam3_model initialized", self.logger.INFO)

    def preprocess_image(self, image):
        orig_h, orig_w = image.shape[:2]
        scale = self.target_size / max(orig_h, orig_w)
        new_h = int(orig_h * scale)
        new_w = int(orig_w * scale)
        resized_image = cv2.resize(image, (new_w, new_h)) # resize image without distorting
        normalized_image = (resized_image - self.pixel_mean) / self.pixel_std
        
        pad_h = self.target_size - new_h
        pad_w = self.target_size - new_w

        padded_image = cv2.copyMakeBorder( # add pad 
            normalized_image,
            top = 0,
            bottom = pad_h,
            left = 0,
            right = pad_w,
            borderType = cv2.BORDER_CONSTANT, 
            value = (0, 0, 0)
        )

        # HWC -> CHW
        input_image = padded_image.transpose(2, 0, 1).astype(np.float32)

        return input_image, (orig_h, orig_w), scale, (new_h, new_w)
    
    def postprocess_mask(self, mask, orig_dims, scale, new_dims):
        # 250 * 250 -> 1008 * 1008 -> original image size
        upscale_mask = cv2.resize(
            mask, 
            (self.target_size, self.target_size),
            interpolation = cv2.INTER_LINEAR
        )

        new_h, new_w = new_dims
        cropped_mask = upscale_mask[:new_h, :new_w] # remove pad 
        
        orig_h, orig_w = orig_dims
        final_mask = cv2.resize( # resize back to original image
            cropped_mask, 
            (orig_w, orig_h), 
            interpolation = cv2.INTER_LINEAR
        )

        return (final_mask > 0.0).astype(np.float32)

    async def process_single_request(self, request):
        try:
            t_start = time.time()
            raw_image_batched = pb_utils.get_input_tensor_by_name(request, "RAW_IMAGE").as_numpy() # get image tensor
            text_prompt_tensor = pb_utils.get_input_tensor_by_name(request, "TEXT_PROMPT").as_numpy() # get text prompt

            b_size = raw_image_batched.shape[0]
            processed_images = []
            prompts = []
            meta = []
            
            for b in range(b_size):
                raw_image = raw_image_batched[b]
                prompt_val = text_prompt_tensor[b]
                prompt_str = prompt_val.item().decode("utf-8") if isinstance(prompt_val, np.ndarray) else str(prompt_val)

                p_img, orig_dims, scale, new_dims = self.preprocess_image(raw_image)
                processed_images.append(p_img)
                prompts.append(prompt_str)
                meta.append((orig_dims, scale, new_dims))

            batched_image = np.stack(processed_images, axis = 0) # group them back up
            
            tokenized = self.tokenizer(
                prompts,
                padding = "max_length",
                max_length = 32,
                truncation = True,
                return_tensors = "np"
            )

            input_ids = tokenized["input_ids"].astype(np.int64)
            attention_mask = tokenized["attention_mask"].astype(np.int64)
            prompt_mask_bool = attention_mask == 1

            vision_req = pb_utils.InferenceRequest(
                model_name = "vision_encoder",
                requested_output_names = ["fpn_feat_0", "fpn_feat_1", "fpn_feat_2", "fpn_pos_2"],
                inputs = [pb_utils.Tensor("images", batched_image)]
            )
            
            text_req = pb_utils.InferenceRequest(
                model_name = "text_encoder",
                requested_output_names = ["text_features"],
                inputs = [
                    pb_utils.Tensor("input_ids", input_ids),
                    pb_utils.Tensor("attention_mask", attention_mask)
                ]
            )

            # concurrently execute vision and text encoders
            vision_future = vision_req.async_exec()
            text_future = text_req.async_exec()
            vision_res, text_res = await asyncio.gather(vision_future, text_future)

            if vision_res.has_error():
                raise pb_utils.TritonModelException(f"vision_encoder failed: {vision_res.error().message()}")
            if text_res.has_error():
                raise pb_utils.TritonModelException(f"text_encoder failed: {text_res.error().message()}")
                
            text_features_tensor = pb_utils.get_output_tensor_by_name(text_res, "text_features")

            
            mask_req = pb_utils.InferenceRequest(
                model_name = "mask_decoder",
                requested_output_names = ["pred_masks", "pred_logits"],
                inputs = [
                    pb_utils.get_output_tensor_by_name(vision_res, "fpn_feat_0"),
                    pb_utils.get_output_tensor_by_name(vision_res, "fpn_feat_1"),
                    pb_utils.get_output_tensor_by_name(vision_res, "fpn_feat_2"),
                    pb_utils.get_output_tensor_by_name(vision_res, "fpn_pos_2"),
                    pb_utils.Tensor.from_dlpack("prompt_features", text_features_tensor.to_dlpack()),
                    pb_utils.Tensor("prompt_mask", prompt_mask_bool)
                ],
                preferred_memory = self.cpu_mem
            )
            
            mask_res = await mask_req.async_exec() # fire mask decoder
            if mask_res.has_error():
                raise pb_utils.TritonModelException(f"mask_decoder failed: {mask_res.error().message()}")
                
            masks = pb_utils.get_output_tensor_by_name(mask_res, "pred_masks").as_numpy()
            logits = pb_utils.get_output_tensor_by_name(mask_res, "pred_logits").as_numpy()
            
            req_masks = []
            for b in range(b_size):
                orig_dims, scale, new_dims = meta[b]
                best_idx = np.argmax(logits[b])
                primary_mask = masks[b][best_idx]
                final_mask = self.postprocess_mask(primary_mask, orig_dims, scale, new_dims)
                req_masks.append(final_mask)
                
            req_masks_batched = np.stack(req_masks, axis = 0)
            out_tensor = pb_utils.Tensor("SEGMENTATION_MASK", req_masks_batched)
            
            latency = (time.time() - t_start) * 1000
            self.logger.log(f"batch={b_size} latency={latency:.0f}ms", self.logger.VERBOSE)
            return pb_utils.InferenceResponse(output_tensors = [out_tensor])
        except Exception as e:
            self.logger.log(f"request failed: {e}", self.logger.ERROR)
            return pb_utils.InferenceResponse(
                output_tensors = [], error = pb_utils.TritonError(str(e))
            )

    async def execute(self, requests):
        # fire off all requests concurrently
        futures = [self.process_single_request(req) for req in requests]
        responses = await asyncio.gather(*futures) # wait for all requests to finish at once
        return responses

    def finalize(self):
        self.logger.log("sam3_model shutting down", self.logger.INFO)
