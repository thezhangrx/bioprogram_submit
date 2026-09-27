# XGBoost Feature Importance & TreeSHAP Robustness (Whitelist)

> Robustness codes (based on SHAP_SNR): `***` SNR>=2.5, `**` >=1.8, `*` >=1.2, `.` >=0.8

| split_type | environment | cell_line | feature | XGB_Gain | XGB_Weight | XGB_Cover | TreeSHAP | SHAP_SNR | sig |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| single | sequence | labuhn | pos1_A | 0.0737 | 29.0000 | 20.2759 | 0.0053 | 0.6539 |  |
| single | sequence | labuhn | pos1_C | 0.0684 | 21.0000 | 18.1905 | 0.0014 | 0.5152 |  |
| single | sequence | labuhn | pos1_G | 0.0935 | 20.0000 | 27.7500 | 0.0035 | 0.8809 | . |
| single | sequence | labuhn | pos1_T | 0.0766 | 18.0000 | 16.6111 | 0.0024 | 0.8752 | . |
| single | sequence | labuhn | pos2_A | 0.0820 | 21.0000 | 18.2381 | 0.0030 | 0.7764 |  |
| single | sequence | labuhn | pos2_C | 0.0844 | 11.0000 | 21.8182 | 0.0010 | 0.5761 |  |
| single | sequence | labuhn | pos2_G | 0.0855 | 15.0000 | 14.4667 | 0.0015 | 0.6893 |  |
| single | sequence | labuhn | pos2_T | 0.0595 | 10.0000 | 24.4000 | 0.0019 | 0.5626 |  |
| single | sequence | labuhn | pos3_A | 0.0360 | 10.0000 | 17.5000 | 0.0014 | 0.7497 |  |
| single | sequence | labuhn | pos3_C | 0.1163 | 19.0000 | 83.3158 | 0.0100 | 0.7896 |  |
| single | sequence | labuhn | pos3_G | 0.1194 | 8.0000 | 34.3750 | 0.0011 | 0.5555 |  |
| single | sequence | labuhn | pos3_T | 0.1125 | 7.0000 | 51.0000 | 0.0027 | 0.8191 | . |
| single | sequence | labuhn | pos4_A | 0.1321 | 12.0000 | 68.3333 | 0.0047 | 0.7566 |  |
| single | sequence | labuhn | pos4_C | 0.0985 | 10.0000 | 35.6000 | 0.0012 | 0.7811 |  |
| single | sequence | labuhn | pos4_G | 0.1728 | 20.0000 | 53.4500 | 0.0079 | 0.7843 |  |
| single | sequence | labuhn | pos4_T | 0.1591 | 17.0000 | 44.8824 | 0.0071 | 0.8019 | . |
| single | sequence | labuhn | pos5_A | 0.1078 | 10.0000 | 57.4000 | 0.0038 | 0.8021 | . |
| single | sequence | labuhn | pos5_C | 0.1153 | 13.0000 | 63.9231 | 0.0047 | 0.7840 |  |
| single | sequence | labuhn | pos5_G | 0.0742 | 11.0000 | 26.9091 | 0.0013 | 0.6878 |  |
| single | sequence | labuhn | pos5_T | 0.1464 | 8.0000 | 25.0000 | 0.0019 | 0.5237 |  |
| single | sequence | labuhn | pos6_A | 0.1234 | 8.0000 | 33.3750 | 0.0010 | 0.6650 |  |
| single | sequence | labuhn | pos6_C | 0.1312 | 11.0000 | 49.7273 | 0.0031 | 0.7397 |  |
| single | sequence | labuhn | pos6_G | 0.1846 | 16.0000 | 114.3125 | 0.0129 | 0.8130 | . |
| single | sequence | labuhn | pos6_T | 0.1082 | 6.0000 | 28.3333 | 0.0016 | 0.7442 |  |
| single | sequence | labuhn | pos7_A | 0.1654 | 10.0000 | 30.7000 | 0.0022 | 0.6710 |  |
| single | sequence | labuhn | pos7_C | 0.1539 | 24.0000 | 63.7917 | 0.0099 | 0.7955 |  |
| single | sequence | labuhn | pos7_G | 0.1236 | 3.0000 | 11.6667 | 0.0005 | 0.6545 |  |
| single | sequence | labuhn | pos7_T | 0.0918 | 7.0000 | 25.5714 | 0.0019 | 0.6877 |  |
| single | sequence | labuhn | pos8_A | 0.1378 | 20.0000 | 41.7500 | 0.0035 | 0.7648 |  |
| single | sequence | labuhn | pos8_C | 0.1062 | 8.0000 | 23.6250 | 0.0012 | 0.7226 |  |
| single | sequence | labuhn | pos8_G | 0.1101 | 10.0000 | 19.7000 | 0.0014 | 0.7130 |  |
| single | sequence | labuhn | pos8_T | 0.0544 | 8.0000 | 19.1250 | 0.0008 | 0.6600 |  |
| single | sequence | labuhn | pos9_A | 0.0955 | 9.0000 | 28.0000 | 0.0017 | 0.5623 |  |
| single | sequence | labuhn | pos9_C | 0.1393 | 15.0000 | 66.4667 | 0.0066 | 0.8624 | . |
| single | sequence | labuhn | pos9_G | 0.1111 | 4.0000 | 28.2500 | 0.0008 | 0.7990 |  |
| single | sequence | labuhn | pos9_T | 0.1532 | 7.0000 | 22.5714 | 0.0009 | 0.5087 |  |
| single | sequence | labuhn | pos10_A | 0.1449 | 14.0000 | 45.3571 | 0.0025 | 0.6951 |  |
| single | sequence | labuhn | pos10_C | 0.0889 | 7.0000 | 24.4286 | 0.0010 | 0.8365 | . |
| single | sequence | labuhn | pos10_G | 0.1188 | 13.0000 | 21.0769 | 0.0035 | 0.8539 | . |
| single | sequence | labuhn | pos10_T | 0.1164 | 11.0000 | 21.2727 | 0.0024 | 0.7578 |  |
| single | sequence | labuhn | pos11_A | 0.1798 | 11.0000 | 39.6364 | 0.0044 | 0.8024 | . |
| single | sequence | labuhn | pos11_C | 0.1721 | 10.0000 | 21.7000 | 0.0017 | 0.6530 |  |
| single | sequence | labuhn | pos11_G | 0.0649 | 7.0000 | 20.0000 | 0.0008 | 0.5910 |  |
| single | sequence | labuhn | pos11_T | 0.1283 | 2.0000 | 42.5000 | 0.0005 | 0.5934 |  |
| single | sequence | labuhn | pos12_A | 0.1773 | 12.0000 | 27.7500 | 0.0021 | 0.7737 |  |
| single | sequence | labuhn | pos12_C | 0.2637 | 4.0000 | 39.0000 | 0.0020 | 0.7761 |  |
| single | sequence | labuhn | pos12_G | 0.1329 | 11.0000 | 56.6364 | 0.0035 | 0.7928 |  |
| single | sequence | labuhn | pos12_T | 0.1770 | 15.0000 | 30.9333 | 0.0038 | 0.6060 |  |
| single | sequence | labuhn | pos13_A | 0.0574 | 5.0000 | 10.0000 | 0.0003 | 0.5977 |  |
| single | sequence | labuhn | pos13_C | 0.1166 | 10.0000 | 41.8000 | 0.0029 | 0.8182 | . |
| single | sequence | labuhn | pos13_G | 0.0976 | 11.0000 | 30.0000 | 0.0008 | 0.6973 |  |
| single | sequence | labuhn | pos13_T | 0.1592 | 15.0000 | 52.4667 | 0.0049 | 0.6933 |  |
| single | sequence | labuhn | pos14_A | 0.1696 | 8.0000 | 60.1250 | 0.0034 | 0.6188 |  |
| single | sequence | labuhn | pos14_C | 0.1831 | 14.0000 | 55.5714 | 0.0049 | 0.7741 |  |
| single | sequence | labuhn | pos14_G | 0.1398 | 10.0000 | 51.7000 | 0.0030 | 0.7383 |  |
| single | sequence | labuhn | pos14_T | 0.1581 | 8.0000 | 38.5000 | 0.0015 | 0.7081 |  |
| single | sequence | labuhn | pos15_A | 0.1967 | 30.0000 | 75.2000 | 0.0164 | 0.8341 | . |
| single | sequence | labuhn | pos15_C | 0.0833 | 8.0000 | 36.5000 | 0.0014 | 0.8246 | . |
| single | sequence | labuhn | pos15_G | 0.1018 | 8.0000 | 24.8750 | 0.0013 | 0.7419 |  |
| single | sequence | labuhn | pos15_T | 0.2302 | 2.0000 | 18.5000 | 0.0006 | 0.7912 |  |
| single | sequence | labuhn | pos16_A | 0.1600 | 5.0000 | 23.0000 | 0.0007 | 0.6367 |  |
| single | sequence | labuhn | pos16_C | 0.0732 | 8.0000 | 13.8750 | 0.0006 | 0.5865 |  |
| single | sequence | labuhn | pos16_G | 0.1562 | 21.0000 | 36.9048 | 0.0072 | 0.8118 | . |
| single | sequence | labuhn | pos16_T | 0.0958 | 4.0000 | 35.5000 | 0.0012 | 0.7066 |  |
| single | sequence | labuhn | pos17_A | 0.1240 | 3.0000 | 16.6667 | 0.0003 | 0.4096 |  |
| single | sequence | labuhn | pos17_C | 0.1995 | 13.0000 | 36.6154 | 0.0034 | 0.7550 |  |
| single | sequence | labuhn | pos17_G | 0.1609 | 8.0000 | 40.8750 | 0.0027 | 0.8149 | . |
| single | sequence | labuhn | pos17_T | 0.1051 | 7.0000 | 32.1429 | 0.0009 | 0.6530 |  |
| single | sequence | labuhn | pos18_A | 0.1625 | 6.0000 | 39.6667 | 0.0016 | 0.7370 |  |
| single | sequence | labuhn | pos18_C | 0.1766 | 20.0000 | 38.5000 | 0.0092 | 0.8169 | . |
| single | sequence | labuhn | pos18_G | 0.3705 | 8.0000 | 64.7500 | 0.0063 | 0.6580 |  |
| single | sequence | labuhn | pos18_T | 0.1046 | 4.0000 | 20.2500 | 0.0006 | 0.4998 |  |
| single | sequence | labuhn | pos19_A | 0.1783 | 5.0000 | 17.8000 | 0.0007 | 0.5122 |  |
| single | sequence | labuhn | pos19_C | 0.1805 | 9.0000 | 14.1111 | 0.0017 | 0.6294 |  |
| single | sequence | labuhn | pos19_G | 0.1413 | 9.0000 | 22.5556 | 0.0011 | 0.4746 |  |
| single | sequence | labuhn | pos19_T | 0.1470 | 21.0000 | 31.2857 | 0.0035 | 0.6097 |  |
| single | sequence | labuhn | pos20_A | 0.2670 | 4.0000 | 32.2500 | 0.0014 | 0.6937 |  |
| single | sequence | labuhn | pos20_C | 0.2263 | 19.0000 | 134.0000 | 0.0202 | 0.8679 | . |
| single | sequence | labuhn | pos20_G | 0.2447 | 16.0000 | 114.9375 | 0.0127 | 0.9824 | . |
| single | sequence | labuhn | pos20_T | 0.0660 | 5.0000 | 16.2000 | 0.0009 | 0.5563 |  |
| single | sequence | labuhn | pos21_A | 0.1795 | 7.0000 | 14.8571 | 0.0020 | 0.8004 | . |
| single | sequence | labuhn | pos21_C | 0.0788 | 8.0000 | 27.3750 | 0.0013 | 0.6306 |  |
| single | sequence | labuhn | pos21_G | 0.0847 | 5.0000 | 27.6000 | 0.0006 | 0.5915 |  |
| single | sequence | labuhn | pos21_T | 0.1540 | 3.0000 | 31.0000 | 0.0010 | 0.6974 |  |
| single | sequence | labuhn | pos22_A | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |  |
| single | sequence | labuhn | pos22_C | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |  |
| single | sequence | labuhn | pos22_G | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |  |
| single | sequence | labuhn | pos22_T | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |  |
| single | sequence | labuhn | pos23_A | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |  |
| single | sequence | labuhn | pos23_C | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |  |
| single | sequence | labuhn | pos23_G | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |  |
| single | sequence | labuhn | pos23_T | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |  |
