import numpy as np

def chromatic_adaptation_transform(image_bgr, method="grey_world", options=None):
    data = image_bgr
    match method:
        case "grey_world":
            mean_bgr = data.mean((0,1))
            mean_abs = mean_bgr.mean()

            data_cat = (data * (mean_abs / mean_bgr)).round().clip(0, 255).astype(np.uint8)

            print(mean_bgr, mean_abs / mean_bgr)
            return data_cat
        case "white_patch":
            print(options) # [lower_p, upper_p]
            try:
                lower_p = float(options[0])
            except (TypeError, IndexError) as e:
                print("Failed to set lower percentile:", e)
                print("Defaulting to 95th percentile")
                lower_p = 95
            try:
                upper_p = float(options[1])
            except (TypeError, IndexError) as e:
                print("Failed to set upper percentile:", e)
                print("Defaulting to 99th percentile")
                upper_p = 99
            lower_p = np.clip(lower_p, 0, 100)
            upper_p = np.clip(upper_p, lower_p, 100)

            distance = np.sum(data.astype(np.uint16) ** 2, (2))
            lower, upper = np.percentile(distance, [lower_p, upper_p])
            print(lower, upper)
            indices = np.nonzero((distance >= lower) * (distance <= upper))
            
            top_bgr = np.mean(data[indices], (0))
            mean_abs = top_bgr.mean()

            data_cat = (data * (mean_abs / top_bgr)).round().clip(0, 255).astype(np.uint8)

            print(top_bgr, mean_abs / top_bgr)

            return data_cat
        case _:
            raise ValueError("Invalid method.")
        
