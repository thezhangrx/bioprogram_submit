# XGBoost Feature Importance & TreeSHAP Robustness (Whitelist)

> Robustness codes (based on SHAP_SNR): `***` SNR>=2.5, `**` >=1.8, `*` >=1.2, `.` >=0.8

| split_type | environment | cell_line | feature | XGB_Gain | XGB_Weight | XGB_Cover | TreeSHAP | SHAP_SNR | sig |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| single | sequence | hiranniramol | pos1_A | 0.0565 | 88.0000 | 99.1932 | 0.0021 | 0.7020 |  |
| single | sequence | hiranniramol | pos1_C | 0.0843 | 117.0000 | 117.2137 | 0.0123 | 0.8181 | . |
| single | sequence | hiranniramol | pos1_G | 0.1155 | 124.0000 | 112.2097 | 0.0129 | 0.8010 | . |
| single | sequence | hiranniramol | pos1_T | 0.1231 | 113.0000 | 113.9380 | 0.0104 | 0.7569 |  |
| single | sequence | hiranniramol | pos2_A | 0.0775 | 79.0000 | 54.1772 | 0.0018 | 0.7133 |  |
| single | sequence | hiranniramol | pos2_C | 0.0939 | 128.0000 | 111.2266 | 0.0112 | 0.8019 | . |
| single | sequence | hiranniramol | pos2_G | 0.0795 | 79.0000 | 69.6962 | 0.0017 | 0.7354 |  |
| single | sequence | hiranniramol | pos2_T | 0.1360 | 96.0000 | 116.5729 | 0.0117 | 0.8025 | . |
| single | sequence | hiranniramol | pos3_A | 0.0788 | 81.0000 | 78.2222 | 0.0038 | 0.7113 |  |
| single | sequence | hiranniramol | pos3_C | 0.0907 | 100.0000 | 110.1300 | 0.0089 | 0.7989 |  |
| single | sequence | hiranniramol | pos3_G | 0.0647 | 101.0000 | 62.4752 | 0.0029 | 0.6767 |  |
| single | sequence | hiranniramol | pos3_T | 0.0625 | 56.0000 | 110.8036 | 0.0019 | 0.7075 |  |
| single | sequence | hiranniramol | pos4_A | 0.0656 | 90.0000 | 136.6444 | 0.0062 | 0.7723 |  |
| single | sequence | hiranniramol | pos4_C | 0.1123 | 91.0000 | 85.1319 | 0.0074 | 0.7519 |  |
| single | sequence | hiranniramol | pos4_G | 0.1320 | 104.0000 | 121.4615 | 0.0080 | 0.7459 |  |
| single | sequence | hiranniramol | pos4_T | 0.0959 | 114.0000 | 122.5000 | 0.0103 | 0.7842 |  |
| single | sequence | hiranniramol | pos5_A | 0.0791 | 110.0000 | 69.8364 | 0.0029 | 0.6553 |  |
| single | sequence | hiranniramol | pos5_C | 0.0741 | 70.0000 | 64.4714 | 0.0030 | 0.7423 |  |
| single | sequence | hiranniramol | pos5_G | 0.0651 | 75.0000 | 77.7867 | 0.0022 | 0.6503 |  |
| single | sequence | hiranniramol | pos5_T | 0.0700 | 69.0000 | 94.5362 | 0.0038 | 0.6562 |  |
| single | sequence | hiranniramol | pos6_A | 0.1369 | 85.0000 | 115.3412 | 0.0067 | 0.7313 |  |
| single | sequence | hiranniramol | pos6_C | 0.0770 | 91.0000 | 129.5604 | 0.0049 | 0.7597 |  |
| single | sequence | hiranniramol | pos6_G | 0.1553 | 100.0000 | 125.6400 | 0.0125 | 0.8271 | . |
| single | sequence | hiranniramol | pos6_T | 0.1522 | 97.0000 | 118.0515 | 0.0133 | 0.8069 | . |
| single | sequence | hiranniramol | pos7_A | 0.0860 | 98.0000 | 100.2755 | 0.0061 | 0.7277 |  |
| single | sequence | hiranniramol | pos7_C | 0.0801 | 91.0000 | 66.5934 | 0.0028 | 0.7041 |  |
| single | sequence | hiranniramol | pos7_G | 0.1145 | 81.0000 | 112.7284 | 0.0074 | 0.7964 |  |
| single | sequence | hiranniramol | pos7_T | 0.1394 | 118.0000 | 124.3644 | 0.0083 | 0.7452 |  |
| single | sequence | hiranniramol | pos8_A | 0.1236 | 79.0000 | 145.7468 | 0.0083 | 0.7726 |  |
| single | sequence | hiranniramol | pos8_C | 0.0940 | 57.0000 | 86.5614 | 0.0020 | 0.6705 |  |
| single | sequence | hiranniramol | pos8_G | 0.1048 | 79.0000 | 92.9620 | 0.0034 | 0.6610 |  |
| single | sequence | hiranniramol | pos8_T | 0.0810 | 94.0000 | 122.0532 | 0.0056 | 0.7183 |  |
| single | sequence | hiranniramol | pos9_A | 0.0783 | 65.0000 | 81.6462 | 0.0029 | 0.7159 |  |
| single | sequence | hiranniramol | pos9_C | 0.0881 | 87.0000 | 97.4253 | 0.0044 | 0.7512 |  |
| single | sequence | hiranniramol | pos9_G | 0.0713 | 80.0000 | 110.4375 | 0.0031 | 0.6674 |  |
| single | sequence | hiranniramol | pos9_T | 0.0736 | 81.0000 | 106.9630 | 0.0023 | 0.7197 |  |
| single | sequence | hiranniramol | pos10_A | 0.0748 | 65.0000 | 142.6000 | 0.0045 | 0.7243 |  |
| single | sequence | hiranniramol | pos10_C | 0.0873 | 66.0000 | 91.3182 | 0.0035 | 0.8105 | . |
| single | sequence | hiranniramol | pos10_G | 0.0682 | 90.0000 | 104.6556 | 0.0069 | 0.8111 | . |
| single | sequence | hiranniramol | pos10_T | 0.1670 | 110.0000 | 186.1455 | 0.0241 | 0.8053 | . |
| single | sequence | hiranniramol | pos11_A | 0.2650 | 89.0000 | 129.7528 | 0.0201 | 0.8021 | . |
| single | sequence | hiranniramol | pos11_C | 0.0809 | 91.0000 | 108.4286 | 0.0048 | 0.7332 |  |
| single | sequence | hiranniramol | pos11_G | 0.0672 | 102.0000 | 98.6569 | 0.0030 | 0.6575 |  |
| single | sequence | hiranniramol | pos11_T | 0.2146 | 116.0000 | 138.7759 | 0.0248 | 0.8304 | . |
| single | sequence | hiranniramol | pos12_A | 0.1497 | 103.0000 | 133.3689 | 0.0113 | 0.7566 |  |
| single | sequence | hiranniramol | pos12_C | 0.0758 | 73.0000 | 113.6712 | 0.0025 | 0.7211 |  |
| single | sequence | hiranniramol | pos12_G | 0.0645 | 75.0000 | 102.7333 | 0.0036 | 0.7636 |  |
| single | sequence | hiranniramol | pos12_T | 0.1012 | 72.0000 | 130.8750 | 0.0063 | 0.7688 |  |
| single | sequence | hiranniramol | pos13_A | 0.1033 | 90.0000 | 81.2444 | 0.0028 | 0.6648 |  |
| single | sequence | hiranniramol | pos13_C | 0.1336 | 112.0000 | 121.9911 | 0.0074 | 0.6706 |  |
| single | sequence | hiranniramol | pos13_G | 0.0814 | 81.0000 | 114.2469 | 0.0033 | 0.7204 |  |
| single | sequence | hiranniramol | pos13_T | 0.1037 | 92.0000 | 104.7609 | 0.0076 | 0.7848 |  |
| single | sequence | hiranniramol | pos14_A | 0.2698 | 89.0000 | 150.8539 | 0.0174 | 0.8441 | . |
| single | sequence | hiranniramol | pos14_C | 0.0799 | 62.0000 | 104.5484 | 0.0023 | 0.7091 |  |
| single | sequence | hiranniramol | pos14_G | 0.1592 | 93.0000 | 143.8495 | 0.0167 | 0.8138 | . |
| single | sequence | hiranniramol | pos14_T | 0.1465 | 93.0000 | 117.3763 | 0.0039 | 0.5393 |  |
| single | sequence | hiranniramol | pos15_A | 0.2196 | 84.0000 | 253.5714 | 0.0232 | 0.7773 |  |
| single | sequence | hiranniramol | pos15_C | 0.0918 | 74.0000 | 162.7838 | 0.0049 | 0.6837 |  |
| single | sequence | hiranniramol | pos15_G | 0.0805 | 95.0000 | 104.2316 | 0.0036 | 0.7145 |  |
| single | sequence | hiranniramol | pos15_T | 0.2037 | 108.0000 | 132.0463 | 0.0134 | 0.8119 | . |
| single | sequence | hiranniramol | pos16_A | 0.1495 | 86.0000 | 168.4419 | 0.0128 | 0.7550 |  |
| single | sequence | hiranniramol | pos16_C | 0.0971 | 94.0000 | 100.3298 | 0.0033 | 0.7198 |  |
| single | sequence | hiranniramol | pos16_G | 0.1201 | 133.0000 | 125.3158 | 0.0113 | 0.7997 |  |
| single | sequence | hiranniramol | pos16_T | 0.2100 | 88.0000 | 131.7159 | 0.0149 | 0.7344 |  |
| single | sequence | hiranniramol | pos17_A | 0.0977 | 88.0000 | 90.4205 | 0.0027 | 0.5965 |  |
| single | sequence | hiranniramol | pos17_C | 0.2399 | 88.0000 | 121.6591 | 0.0180 | 0.8218 | . |
| single | sequence | hiranniramol | pos17_G | 0.1287 | 115.0000 | 73.7565 | 0.0038 | 0.7174 |  |
| single | sequence | hiranniramol | pos17_T | 0.2578 | 118.0000 | 182.7627 | 0.0313 | 0.8017 | . |
| single | sequence | hiranniramol | pos18_A | 0.0750 | 107.0000 | 107.1963 | 0.0026 | 0.6542 |  |
| single | sequence | hiranniramol | pos18_C | 0.2490 | 77.0000 | 172.3117 | 0.0196 | 0.7869 |  |
| single | sequence | hiranniramol | pos18_G | 0.1534 | 108.0000 | 98.8704 | 0.0061 | 0.6906 |  |
| single | sequence | hiranniramol | pos18_T | 0.2561 | 114.0000 | 119.9474 | 0.0222 | 0.8125 | . |
| single | sequence | hiranniramol | pos19_A | 0.0867 | 99.0000 | 114.7879 | 0.0033 | 0.6751 |  |
| single | sequence | hiranniramol | pos19_C | 0.1014 | 104.0000 | 117.6346 | 0.0038 | 0.7067 |  |
| single | sequence | hiranniramol | pos19_G | 0.3793 | 99.0000 | 165.6061 | 0.0372 | 0.8341 | . |
| single | sequence | hiranniramol | pos19_T | 0.5734 | 125.0000 | 180.9040 | 0.0509 | 0.8168 | . |
| single | sequence | hiranniramol | pos20_A | 0.1722 | 100.0000 | 140.3600 | 0.0086 | 0.7411 |  |
| single | sequence | hiranniramol | pos20_C | 0.1757 | 165.0000 | 117.9273 | 0.0155 | 0.7819 |  |
| single | sequence | hiranniramol | pos20_G | 0.6117 | 160.0000 | 171.2937 | 0.0656 | 0.8458 | . |
| single | sequence | hiranniramol | pos20_T | 0.3697 | 93.0000 | 186.0215 | 0.0358 | 0.8272 | . |
| single | sequence | hiranniramol | pos21_A | 0.0958 | 97.0000 | 136.4536 | 0.0098 | 0.7870 |  |
| single | sequence | hiranniramol | pos21_C | 0.1629 | 85.0000 | 126.9529 | 0.0096 | 0.7274 |  |
| single | sequence | hiranniramol | pos21_G | 0.3509 | 89.0000 | 169.3596 | 0.0283 | 0.8298 | . |
| single | sequence | hiranniramol | pos21_T | 0.1457 | 82.0000 | 231.5244 | 0.0171 | 0.7874 |  |
| single | sequence | hiranniramol | pos22_A | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |  |
| single | sequence | hiranniramol | pos22_C | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |  |
| single | sequence | hiranniramol | pos22_G | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |  |
| single | sequence | hiranniramol | pos22_T | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |  |
| single | sequence | hiranniramol | pos23_A | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |  |
| single | sequence | hiranniramol | pos23_C | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |  |
| single | sequence | hiranniramol | pos23_G | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |  |
| single | sequence | hiranniramol | pos23_T | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |  |
