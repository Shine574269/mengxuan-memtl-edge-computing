import json


def store_model(model, path):
    # Save model config as JSON (compatible with Keras 3.x which removed to_json)
    config = model.get_config()
    with open(path + ".json", "w") as json_file:
        json.dump(config, json_file, indent=2)
    # Keras 3.x requires .weights.h5 extension for save_weights
    model.save_weights(path + ".weights.h5")
    print("[store_model] model with {} mu_num stored to {}."
          .format(model.mu_num, path))