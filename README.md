# FedELC
Official codes for CIKM '24 full paper: Tackling Noisy Clients in Federated Learning with End-to-end Label Correction. The Arxiv preprint is available in [here](https://arxiv.org/abs/2408.04301).

## Abstract
Recently, federated learning (FL) has achieved wide successes for diverse privacy-sensitive applications without sacrificing the sensitive private information of clients. However, the data quality of client datasets can not be guaranteed since corresponding annotations of different clients often contain complex label noise of varying degrees, which inevitably causes the performance degradation. Intuitively, the performance degradation is dominated by clients with higher noise rates since their trained models contain more misinformation from data, thus it is necessary to devise an effective optimization scheme to mitigate the negative impacts of these noisy clients. In this work, we propose a two-stage framework FedELC to tackle this complicated label noise issue. The first stage aims to guide the detection of noisy clients with higher label noise, while the second stage aims to correct the labels of noisy clients' data via an end-to-end label correction framework which is achieved by learning possible ground-truth labels of noisy clients' datasets via back propagation. We implement sixteen related methods and evaluate five datasets with three types of complicated label noise scenarios for a comprehensive comparison. Extensive experimental results demonstrate our proposed framework achieves superior performance than its counterparts for different scenarios. Additionally, we effectively improve the data quality of detected noisy clients' local datasets with our label correction framework.

## Codes are coming soon... 

