# AI CUP 2022 Fall NLP

## 運行環境

### 使用 Anaconda 建立環境

```bash
conda env create -f enviroment.yml
conda activate nlp-aicup
```

## 資料

- `data/length`: 存放 id 對 q、r Token 長度的映射表
- `data/multi_target`: 存放 Multi-Target 資料集
- `data/span`: 存放 Span 資料集
- `data/splitted`: 存放切分的 Train、Val、Test 資料集
- `data/full.csv`: 官方提供的訓練資料
- `data/submission.csv`: 官方提供的測試資料

處理腳本

- `scripts/split_dataset.py`: 分割原始資料集
- `scripts/generate_span_dataset.py`: 由原始資料集生成 Span 資料集
- `scripts/generate_mutli_target_dataset.py`: 由原始資料集生成 Multi-Target 資料集

## 訓練

提供訓練腳本來演示各模型架構的訓練

- 分離式 + Token Classification: `train_siamese_token_classification.py`
- 分離式 + Span Prediction: `train_siamese_span_prediction.py`
- 分離式 + Span Prediction + LSTM: `train_siamese_span_prediction_with_lstm.py`
- 合併式 + Token Classification: `train_token_classification.py`
- 合併式 + Span Prediction: `train_span_prediction.py`

## 預測

提供 Jupyter Notebook 來演示各個模型架構進行預測的具體流程

- 分離式 + Span Prediction: `notebooks/predict_span_prediction_model.ipynb`
- 分離式 + Token Classification + LSTM: `notebooks/predict_siamese_span_prediction_model_with_lstm.ipynb`
- 分離式 + Token Classification: `notebooks/predict_siamese_token_classification_model.ipynb`
- 合併式 + Span Prediction: `notebooks/predict_span_prediction_model.ipynb`
- 合併式 + Token Classification: `notebooks/predict_token_classification_model.ipynb`

Span Prediction 模型在預測時可以調整以下參數
- top_k: 考慮機率最大的 K 個 Span
- max_tokens: 每個 Span 最多可以包含幾個 Token

## 重要模組介紹

Answer 格式: 型態為 Dict，格式如下所示，用來表示一筆資料的答案(q'、r')
```python3
{
    'id': '...',
    'q': '...',
    'r': '...'
}
```

### 評分模組： `src/scoring`

此模組實現官方提供的計分公式，用來初步評估模型效能

|    Function     | Input                                 | Output   |
| :-------------: | :------------------------------------ | :------- |
| `compute_score` | 輸入一個 List，內容為 Answer          | 輸出評分 |
|  `compute_csv`  | 輸入一個 CSV 檔案路徑，格式同官方規定 | 輸出評分 |

### 資料模組： `src/data`

|        Class         | Description                          | Input File                                 |
| :------------------: | :----------------------------------- | ------------------------------------------ |
|     `RawDataset`     | 用來讀取原始資料集                   | `data/full.csv` <br> `data/splitted/*.csv` |
| `PredictionDataset`  | 預測時使用此資料集進行讀取及前處理   | `data/full.csv` <br> `data/splitted/*.csv` |
|    `SpanDataset`     | 用來讀取 Span 資料集及前處理         | `data/span/*.jsonl`                        |
| `MultiTargetDataset` | 用來讀取 Multi-Target 資料集及前處理 | `data/multi_target/*.jsonl`                |

上表所列的 Class 僅供**合併式**的模型使用，而**分離式**的模型需使用後綴為 `ForSiamese` 的 Class，如：分離式 + TokenClassification 的模型需使用 `MultiTargetDatasetForSiamese` 進行訓練，並使用 `PredictionDatasetForSiamese` 進行預測

### 模型模組： `src/model`

|                Class                 | Description                     |   Dataset Class For Training   | Dataset Class For Predicting  |
| :----------------------------------: | :------------------------------ | :----------------------------: | :---------------------------: |
|        `SpanPredictionModel`         | 合併式 + Span Prediction        |         `SpanDataset`          |      `PredictionDataset`      |
|     `SiameseSpanPredictionModel`     | 分離式 + Span Prediction        |    `SpanDatasetForSiamese`     | `PredictionDatasetForSiamese` |
| `SiameseSpanPredictionModelWithLSTM` | 分離式 + Span Prediction + LSTM |    `SpanDatasetForSiamese`     | `PredictionDatasetForSiamese` |
|      `TokenClassificationModel`      | 合併式 + Token Classification   |      `MultiTargetDataset`      |      `PredictionDataset`      |
|  `SiameseTokenClassificationModel`   | 分離式 + Token Classification   | `MultiTargetDatasetForSiamese` | `PredictionDatasetForSiamese` |

### 後處理相關： `src/utils.py`

|       Function       | Description                                                                                    |
| :------------------: | :--------------------------------------------------------------------------------------------- |
|   `write_answers`    | 輸出官方規定上傳格式的檔案                                                                     |
| `select_starts_ends` | Span Prediction 的後處理函數之一，只先對 Logits 做初步處理，主要邏輯在 `decode_spans` 和 `nms` |
|    `decode_spans`    | Span Prediction 的後處理函數之一，計算 Span 機率，並過濾無效的 Span                            |
|        `nms`         | Span Prediction 的後處理函數之一，對 `decode_spans` 的結果進行 NMS 處理                        |
|  `indices_to_spans`  | 將連續的 index 轉為 Span，用於 Token Classification 的後處理函數                               |

## 模型權重

|           Model Type            |   Epoch   | Public Score | Private Score | URL                                                                                                     |
| :-----------------------------: | :-------: | :----------: | :-----------: | :------------------------------------------------------------------------------------------------------ |
| 分離式 + Span Prediction + LSTM | 1 (full)  |   0.836781   | **0.898072**  | [Download](https://github.com/ShinoharaHare/AI-CUP-2022-Fall-NLP/releases/download/v0.0.0/s-sp-lstm.pt) |
|    合併式 + Span Prediction     | 1 (full)  | **0.845039** |   0.895501    | [Download](https://github.com/ShinoharaHare/AI-CUP-2022-Fall-NLP/releases/download/v0.0.0/sp.pt)        |
|    分離式 + Span Prediction     | 2 (full)  |   0.842063   |   0.890389    | 因遭誤刪，無法提供此權重                                                                                |
|    分離式 + Span Prediction     | 1 (full)  |   0.836294   |   0.887342    | [Download](https://github.com/ShinoharaHare/AI-CUP-2022-Fall-NLP/releases/download/v0.0.0/s-sp.pt)      |
|  分離式 + Token Classification  | 2 (train) |      x       |       x       | [Download](https://github.com/ShinoharaHare/AI-CUP-2022-Fall-NLP/releases/download/v0.0.0/s-tc.pt)      |
|  合併式 + Token Classification  | 2 (train) |      x       |       x       | [Download](https://github.com/ShinoharaHare/AI-CUP-2022-Fall-NLP/releases/download/v0.0.0/tc.pt)        |

註：為節省儲存空間，在進行實驗時我們會定期清理一些權重，因此部分權重並沒有保留下來。上表所提供的部分權重可能並非與當初完全相同，而是由後來進行重新訓練所得到，所以部分權重的分數可能與上表所示不相同
