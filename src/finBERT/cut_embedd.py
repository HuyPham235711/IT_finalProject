import numpy as np

finbert = np.load("E:/TDTu/TAI_LIEU/KY1-NAM5/DU_AN_CNTT/results/finbert/sampler_v2/finbert_daily_embeddings_test.npy")
finbert_cut = finbert[:3589]
np.save("E:/TDTu/TAI_LIEU/KY1-NAM5/DU_AN_CNTT/results/finbert/sampler_v2/finbert_daily_embeddings_test_cut.npy", finbert_cut)
print(finbert_cut.shape)
