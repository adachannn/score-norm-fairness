MODELS={
  "E1": "casia_resnet34_arcface",
  "E2": "ms1m_wo_rfw_resnet50_arcface",
  "E3": "webface12m_iresnet100_adaface",
  "E4" : "ms1m_iresnet100_magface",
  "E5": "lr_iresnet100_arcface"
}

def feature_directory(dataset,data_directory,model):
  import os
  return os.path.join(data_directory, f"{dataset}_{MODELS[model]}")
