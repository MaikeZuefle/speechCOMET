from utils import Data

def run(data: Data) -> list[float]:
    import comet
    model_path = comet.download_model("zouharvi/COMET-partial")
    model = comet.load_from_checkpoint(model_path)
    return model.predict([
        {
            "src": line["src_text"],
            "mt": line["tgt_text"],
        }
        for line in data
    ]).scores