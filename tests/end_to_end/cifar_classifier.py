#**************************************************************************
#||                        SiMa.ai CONFIDENTIAL                          ||
#||   Unpublished Copyright (c) 2024 SiMa.ai, All Rights Reserved.       ||
#**************************************************************************
# NOTICE:  All information contained herein is, and remains the property of
# SiMa.ai. The intellectual and technical concepts contained herein are
# proprietary to SiMa and may be covered by U.S. and Foreign Patents,
# patents in process, and are protected by trade secret or copyright law.
#
# Dissemination of this information or reproduction of this material is
# strictly forbidden unless prior written permission is obtained from
# SiMa.ai.  Access to the source code contained herein is hereby forbidden
# to anyone except current SiMa.ai employees, managers or contractors who
# have executed Confidentiality and Non-disclosure agreements explicitly
# covering such access.
#
# The copyright notice above does not evidence any actual or intended
# publication or disclosure  of  this source code, which includes information
# that is confidential and/or proprietary, and is a trade secret, of SiMa.ai.
#
# ANY REPRODUCTION, MODIFICATION, DISTRIBUTION, PUBLIC PERFORMANCE, OR PUBLIC
# DISPLAY OF OR THROUGH USE OF THIS SOURCE CODE WITHOUT THE EXPRESS WRITTEN
# CONSENT OF SiMa.ai IS STRICTLY PROHIBITED, AND IN VIOLATION OF APPLICABLE
# LAWS AND INTERNATIONAL TREATIES. THE RECEIPT OR POSSESSION OF THIS SOURCE
# CODE AND/OR RELATED INFORMATION DOES NOT CONVEY OR IMPLY ANY RIGHTS TO
# REPRODUCE, DISCLOSE OR DISTRIBUTE ITS CONTENTS, OR TO MANUFACTURE, USE, OR
# SELL ANYTHING THAT IT  MAY DESCRIBE, IN WHOLE OR IN PART.
#
#**************************************************************************
import os
import logging
import argparse
from argparse import Namespace
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from functools import reduce
from operator import or_

from tqdm import tqdm
import numpy as np

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import CIFAR10

from torch import optim, nn, utils, Tensor
from torch.nn import CrossEntropyLoss

from torch.fx.graph_module import GraphModule

import onnxruntime

import pytorch_lightning as L

from sima_qat.qat_api import sima_prepare_qat_model, sima_finalize_qat_model, sima_export_onnx


class CIFAR10Mini(CIFAR10):
    """ A distilled version of CIFAR10 which uses k-means clustering on image embeddings to choose
        good samples of the original dataset. The distillation code is in misc/distill_cifar10.py.

        This distilled verison uses 250 samples per class (=2500 samples) for the train set, and
        50 samples per class (=500 samples) for the test set.
    """
    # 100-sample variant
    # train_idx_map = np.array([779, 1112, 279, 22, 93, 192, 18, 80, 7530, 2933, 308, 493, 367, 503, 107, 858, 1410, 2285, 2053, 799, 158, 115, 830, 3766, 3314, 531, 467, 2351, 5753, 3106, 274, 5743, 2059, 723, 383, 206, 299, 5005, 1468, 709, 171, 3667, 2595, 734, 3403, 300, 148, 205, 2727, 3143, 352, 4402, 2816, 2683, 273, 245, 419, 312, 2342, 3840, 102, 61, 2329, 2757, 95, 12, 2295, 2918, 1719, 2110, 838, 2, 97, 577, 5849, 477, 195, 136, 1173, 90, 729, 907, 1083, 2025, 2363, 248, 1119, 10577, 133, 783, 135, 636, 190, 6192, 483, 1161, 2264, 3807, 921, 32, 2422, 598, 309, 1015, 160, 167, 572, 1344, 1454, 60, 782, 879, 1384, 1338, 29, 941, 1692, 114, 143, 518, 26, 4645, 261, 125, 398, 439, 290, 738, 3212, 284, 721, 1886, 1721, 55, 48, 549, 7920, 787, 831, 344, 1207, 678, 1403, 405, 1193, 1114, 394, 1017, 250, 755, 2517, 1039, 1950, 667, 1189, 240, 1228, 1373, 695, 2953, 111, 4299, 955, 52, 1750, 644, 1301, 386, 50, 1346, 1259, 321, 1300, 568, 124, 193, 566, 214, 1292, 1660, 36, 1154, 1616, 968, 231, 922, 1192, 637, 951, 1049, 888, 211, 1159, 297, 3167, 324, 822, 2583, 720, 239, 1224, 374, 1355, 590, 1828, 2716, 241, 609, 1125, 49, 760, 2132, 742, 3608, 1707, 86, 2718, 215, 5817, 4853, 551, 1538, 132, 340, 44, 331, 1446, 229, 1598, 5721, 397, 1495, 627, 20733, 911, 608, 1033, 616, 3776, 353, 318, 87, 366, 2540, 81, 286, 5000, 2207, 759, 1493, 1773, 959, 1260, 221, 212, 2081, 450, 4520, 1045, 726, 575, 68, 3495, 679, 271, 2605, 841, 2673, 3866, 2311, 1266, 2485, 430, 843, 8532, 56, 7503, 1037, 774, 423, 220, 38577, 1287, 120, 282, 826, 400, 478, 2506, 1550, 1755, 532, 1134, 3970, 2639, 833, 492, 479, 39, 1362, 4606, 589, 1265, 276, 1421, 790, 157, 489, 1043, 110, 1325, 887, 2228, 113, 1628, 79, 3709, 6872, 812, 1394, 350, 30, 594, 1874, 1295, 5095, 486, 1223, 1270, 1817, 5088, 867, 1520, 109, 789, 1864, 1104, 71, 199, 1509, 127, 705, 174, 351, 597, 1543, 1927, 3370, 4391, 2996, 291, 846, 485, 946, 112, 1789, 593, 3371, 587, 1304, 1718, 521, 435, 426, 126, 541, 1234, 10287, 3189, 528, 182, 384, 156, 663, 2548, 3, 805, 1255, 1235, 556, 365, 680, 540, 550, 2397, 332, 542, 4, 247, 361, 451, 69, 828, 1380, 4405, 173, 58, 1229, 1450, 629, 1383, 4312, 2640, 638, 923, 5032, 1368, 2083, 2811, 2006, 513, 263, 328, 207, 201, 310, 725, 621, 5871, 187, 1060, 357, 1354, 372, 2474, 1670, 82, 3725, 433, 10, 1787, 1321, 839, 89, 948, 77, 99, 3960, 3953, 2581, 650, 230, 578, 346, 388, 147, 185, 209, 3487, 632, 2825, 390, 536, 155, 2141, 380, 5448, 96, 642, 462, 1590, 553, 236, 1544, 703, 2766, 391, 258, 2419, 1076, 814, 3165, 820, 188, 122, 1399, 184, 452, 3838, 2125, 1357, 1317, 1408, 164, 53, 1555, 1151, 25, 223, 2034, 687, 635, 191, 165, 1181, 319, 662, 2938, 130, 836, 85, 672, 574, 325, 5, 425, 475, 1798, 1257, 6605, 794, 402, 436, 437, 140, 913, 1041, 718, 1040, 4160, 1432, 1099, 2675, 1187, 613, 1494, 23, 889, 1890, 259, 584, 265, 747, 692, 228, 4229, 586, 116, 293, 31, 170, 1072, 848, 1820, 373, 445, 527, 4752, 4775, 543, 1420, 622, 264, 1776, 810, 6, 2219, 253, 1613, 714, 1764, 427, 852, 76, 649, 401, 2051, 1013, 129, 1205, 224, 235, 413, 448, 9975, 840, 1703, 288, 710, 2088, 180, 38, 2156, 381, 2446, 320, 1705, 298, 962, 770, 83, 139, 1238, 15, 610, 658, 1129, 971, 506, 242, 1882, 1226, 1533, 498, 3589, 490, 395, 141, 5166, 900, 1983, 1324, 181, 2375, 1634, 277, 2109, 73, 768, 117, 735, 1289, 1850, 305, 101, 876, 1011, 1271, 523, 333, 1263, 121, 2121, 1268, 3754, 1244, 3983, 17, 732, 1313, 778, 7012, 11341, 819, 634, 750, 176, 1146, 457, 1458, 2659, 334, 547, 614, 9, 1775, 1027, 829, 940, 443, 1857, 781, 1378, 2100, 466, 183, 3580, 1106, 387, 11, 698, 2546, 178, 2664, 217, 1006, 5417, 1619, 1691, 131, 529, 859, 4063, 656, 19, 1322, 2007, 990, 2443, 59, 216, 428, 1361, 194, 517, 316, 294, 1127, 252, 1646, 62, 362, 424, 208, 1122, 311, 63, 2160, 473, 3855, 512, 691, 238, 803, 991, 256, 88, 1302, 797, 752, 94, 474, 509, 1526, 432, 1339, 3570, 123, 5877, 1650, 534, 1088, 793, 3632, 1440, 1335, 103, 249, 5965, 1834, 795, 23983, 2379, 524, 1490, 2302, 1429, 7155, 2056, 203, 1182, 243, 28, 558, 24, 1701, 14, 1388, 196, 845, 66, 134, 3630, 100, 4956, 3591, 106, 914, 1, 91, 153, 602, 1438, 422, 285, 1190, 8068, 974, 72, 411, 3716, 296, 2024, 356, 3079, 1239, 210, 67, 564, 255, 8470, 947, 47, 146, 142, 2182, 161, 3680, 5512, 2927, 268, 3639, 303, 1463, 4886, 37, 866, 1485, 1206, 1500, 1178, 480, 514, 2807, 654, 5228, 2310, 159, 410, 21, 306, 661, 666, 582, 909, 2041, 906, 234, 2567, 57, 630, 1258, 2734, 664, 226, 13, 289, 507, 74, 368, 35, 2098, 1897, 329, 246, 1583, 233, 1042, 219, 934, 336, 431, 2281, 11574, 1486, 1977, 33, 1763, 7, 772, 108, 16, 213, 1734, 713, 463, 557, 956, 417, 758, 2451, 681, 525, 2769, 712, 977, 163, 34, 1992, 3936, 1727, 5318, 993, 54, 954, 1989, 3180, 1222, 358, 1002, 555, 715, 20, 2487, 1831, 379, 5684, 1397, 676, 3481, 64, 42, 418, 929, 1073, 75, 162, 916, 1413, 369, 8, 4542, 222, 966, 359, 1428, 791, 251, 3582, 149, 643, 1724, 302, 408, 868, 375, 218, 933, 739, 4632, 699, 8886, 3521, 1768, 853, 767, 416, 43, 84, 2943, 1832, 2632, 392, 883, 3773, 716, 459, 2461, 41, 27, 1600, 807, 1056, 270, 487, 777, 1059, 776, 272, 874, 166, 304, 2466, 2393, 1551, 4615, 1148, 633, 1087, 619, 899, 0, 46, 145, 65, 269, 896, 1512, 545, 1472, 40, 342, 315, 225, 45, 78, 202, 1365, 1654, 697, 972, 1046, 842, 267, 1710, 737, 1740, 786, 624, 168])
    # test_idx_map = np.array([74, 854, 111, 90, 258, 406, 3, 179, 235, 44, 352, 206, 315, 97, 52, 287, 10, 297, 27, 281, 1363, 81, 261, 104, 6, 37, 351, 240, 246, 290, 540, 131, 305, 871, 201, 114, 105, 9, 82, 204, 75, 160, 123, 35, 129, 149, 84, 219, 70, 448, 450, 25, 67, 387, 138, 113, 65, 1110, 374, 630, 46, 273, 336, 277, 882, 279, 53, 106, 91, 271, 115, 63, 456, 0, 1163, 176, 608, 103, 8, 565, 100, 223, 32, 26, 682, 500, 40, 1526, 505, 130, 159, 930, 278, 295, 314, 58, 606, 36, 22, 466, 42, 85, 24, 12, 181, 190, 39, 128, 141, 510, 238, 1019, 16, 525, 262, 343, 33, 319, 207, 31, 140, 59, 30, 329, 4, 7, 542, 19, 163, 49, 747, 93, 429, 512, 62, 29, 162, 142, 590, 5, 220, 521, 942, 268, 620, 99, 20, 109, 119, 48, 445, 203, 419, 87, 137, 194, 17, 83, 56, 13, 242, 79, 55, 15, 218, 519, 1, 132, 72, 233, 472, 108, 51, 2, 54, 92, 150, 265, 683, 80, 38, 139, 47, 76, 400, 680, 151, 411, 222, 23, 532, 175, 170, 331, 133, 11, 28, 45, 34, 157])
    # 250-sample variant
    train_idx_map = np.array([1684, 223, 4935, 1980, 637, 5094, 1470, 1664, 4577, 7490, 6650, 1524, 2401, 3852, 35, 7470, 3361, 2490, 628, 1607, 1935, 2895, 700, 1601, 352, 843, 626, 220, 6920, 600, 2248, 417, 2107, 812, 77, 165, 3292, 415, 341, 5814, 1144, 2692, 3515, 783, 348, 401, 276, 2355, 3438, 4061, 8774, 4163, 2959, 4592, 1329, 12009, 49, 332, 1711, 9401, 93, 2863, 1514, 11912, 8660, 4778, 4952, 1466, 9755, 457, 467, 1589, 7053, 129, 7299, 2262, 555, 2233, 4162, 5115, 3441, 1999, 9983, 12129, 5003, 3066, 4314, 504, 965, 4957, 1306, 2773, 3334, 4192, 757, 1188, 1243, 1260, 8268, 2169, 30, 308, 6373, 9325, 8655, 698, 598, 3344, 5477, 6663, 115, 4683, 2034, 608, 1178, 1097, 2342, 4721, 7263, 1668, 405, 5314, 2680, 213, 1039, 3224, 2391, 1296, 1211, 1626, 10601, 5533, 1674, 9140, 1340, 4559, 453, 284, 1266, 527, 2950, 6132, 1338, 4387, 989, 5010, 5657, 1400, 1187, 1885, 3347, 5346, 1216, 4523, 1382, 799, 3582, 5249, 650, 5516, 1185, 694, 782, 8200, 2996, 2413, 2429, 822, 2613, 1594, 2859, 557, 4571, 713, 3938, 1278, 497, 376, 564, 468, 1632, 10399, 687, 1926, 293, 1871, 4540, 2403, 407, 6512, 439, 5924, 1195, 1130, 1709, 448, 708, 1454, 7655, 264, 2512, 974, 2855, 752, 4165, 5757, 2659, 404, 9596, 9253, 2365, 5331, 116, 349, 5468, 179, 1629, 1234, 2804, 1701, 9696, 1424, 3204, 373, 7360, 317, 605, 695, 2145, 927, 2642, 1988, 1012, 2427, 2574, 1872, 344, 199, 185, 1715, 2277, 29, 9058, 189, 1755, 6015, 6954, 481, 4563, 9193, 606, 2802, 9138, 10868, 10627, 4958, 568, 126, 262, 12648, 65, 8307, 5301, 427, 5146, 2185, 6630, 79, 44, 5287, 5224, 75, 7379, 12299, 617, 565, 841, 9400, 2173, 99, 396, 61, 1586, 32, 2037, 304, 4390, 9042, 226, 2762, 3327, 8529, 10188, 2029, 2994, 227, 255, 60, 1006, 96, 917, 11308, 45, 323, 2072, 11007, 4250, 325, 201, 6464, 2094, 9884, 3078, 3064, 6813, 4100, 1304, 833, 11782, 1052, 1731, 1985, 6709, 743, 2582, 1305, 250, 2597, 2101, 978, 1240, 1464, 942, 3924, 12210, 8107, 2455, 8191, 10145, 593, 977, 10153, 1660, 1421, 2615, 4977, 1320, 1287, 119, 364, 302, 7827, 599, 561, 3285, 301, 1565, 2795, 3851, 873, 375, 2200, 2046, 823, 168, 1621, 11767, 6922, 8216, 1242, 6869, 4528, 454, 576, 1574, 753, 236, 2445, 5714, 1029, 10393, 1724, 140, 2771, 4299, 5392, 5, 1410, 8232, 2756, 1604, 97, 212, 257, 4953, 4, 6120, 2775, 980, 4631, 160, 3744, 1251, 6456, 7288, 2320, 1037, 354, 1145, 5099, 2769, 1571, 2149, 5459, 7428, 1790, 134, 1547, 4492, 690, 1548, 747, 2186, 5387, 2431, 2630, 5816, 6831, 275, 1707, 493, 3555, 2052, 486, 6659, 6798, 2727, 4172, 1631, 3120, 9717, 2038, 10049, 2469, 46, 3231, 5831, 5112, 184, 8556, 137, 1570, 64, 4039, 578, 482, 1973, 5261, 1869, 835, 206, 461, 1377, 6040, 962, 3661, 1502, 4942, 4444, 389, 105, 997, 8576, 1736, 6703, 311, 4661, 94, 834, 4562, 7640, 3523, 1541, 176, 3009, 4648, 4560, 282, 2841, 4101, 8825, 1064, 1293, 6196, 5076, 136, 11750, 4983, 3233, 312, 3338, 423, 1040, 4113, 7522, 5575, 6542, 1852, 121, 12058, 1110, 1307, 2224, 7745, 2484, 463, 2537, 513, 9404, 6520, 803, 2742, 510, 1800, 2543, 1189, 5576, 778, 3662, 5554, 2080, 2760, 9556, 4806, 271, 796, 1207, 696, 827, 1971, 6704, 3669, 2467, 1981, 2868, 558, 2217, 7106, 1354, 2936, 1139, 1609, 5686, 8736, 2869, 1756, 1008, 737, 4351, 108, 3417, 3423, 2237, 1861, 144, 975, 218, 1295, 1163, 1444, 2044, 2470, 18, 3210, 13, 1641, 3019, 1108, 6019, 3291, 2138, 4748, 1921, 2719, 701, 1532, 1995, 808, 196, 474, 5901, 483, 9900, 1792, 47, 4431, 9299, 1768, 1077, 2557, 1693, 2533, 2708, 2975, 2191, 2927, 1576, 2619, 1291, 194, 910, 1566, 3639, 586, 57, 1844, 90, 42, 885, 2971, 810, 10180, 7589, 2291, 889, 2704, 1990, 288, 303, 986, 2561, 8043, 957, 421, 4634, 2349, 5871, 1884, 400, 912, 11961, 540, 1830, 48, 7308, 3390, 1492, 4398, 6660, 1533, 6816, 6392, 411, 300, 4122, 673, 646, 4830, 559, 2408, 54, 8377, 830, 6616, 3511, 1867, 2491, 10935, 5967, 425, 1372, 1067, 779, 1798, 1752, 1677, 539, 648, 5123, 55, 3037, 1497, 2812, 2789, 383, 8341, 3476, 5762, 5127, 2136, 2041, 6097, 11342, 990, 171, 5933, 2121, 403, 8459, 1409, 1818, 963, 1288, 1219, 6738, 8485, 11415, 820, 3586, 5248, 538, 1727, 123, 2993, 1003, 790, 6210, 2550, 4012, 41, 120, 281, 1883, 4703, 6, 4014, 138, 1180, 2535, 2062, 2098, 283, 649, 8186, 3847, 3416, 907, 7211, 1500, 3119, 4356, 6026, 8411, 5781, 2177, 24, 5535, 1328, 2860, 402, 1966, 2057, 1076, 384, 7566, 11344, 5877, 785, 4414, 3905, 7171, 5221, 1846, 416, 998, 517, 685, 1208, 1888, 6782, 7138, 3003, 370, 1895, 2668, 1252, 8013, 792, 445, 1568, 2521, 3026, 26, 395, 3594, 857, 1257, 1025, 878, 1785, 1351, 2068, 2386, 315, 1059, 342, 7075, 6700, 702, 1710, 3105, 4847, 6071, 1124, 4083, 882, 3178, 377, 4669, 3218, 2564, 2082, 2376, 3376, 4836, 639, 1316, 995, 2197, 5400, 1696, 1539, 6217, 1098, 17, 2770, 1196, 4745, 91, 1286, 2279, 3709, 5423, 2947, 5615, 8262, 7487, 1534, 203, 1427, 59, 774, 10017, 2048, 101, 4827, 9345, 1778, 1573, 2723, 869, 7228, 141, 3320, 479, 1834, 583, 11760, 3289, 241, 6574, 9647, 3286, 1074, 6151, 3807, 7749, 5870, 3052, 3870, 1856, 3391, 895, 3043, 969, 2665, 2581, 253, 266, 1048, 2137, 691, 1658, 2738, 2594, 3795, 6256, 1603, 7893, 9034, 955, 39, 1919, 5963, 5006, 9817, 207, 494, 258, 2383, 4010, 4950, 367, 1737, 5520, 1938, 922, 78, 597, 1803, 314, 2165, 1496, 603, 3910, 38, 197, 2307, 10135, 2198, 6493, 2174, 801, 1951, 7006, 36, 1109, 684, 1815, 2432, 3108, 740, 334, 949, 331, 9183, 229, 6185, 3110, 159, 9, 6466, 1511, 1055, 1546, 21, 5129, 174, 2025, 1965, 169, 1897, 3762, 638, 6961, 8656, 333, 8739, 4405, 142, 2865, 4294, 150, 8193, 1057, 5475, 1487, 74, 5218, 1002, 1879, 629, 9982, 5065, 80, 1555, 5110, 1202, 1426, 3617, 1962, 287, 9841, 8285, 4179, 6405, 33, 3351, 4954, 7593, 1556, 8686, 8598, 6198, 7689, 1100, 3575, 788, 1070, 4437, 1851, 1595, 1813, 10570, 10300, 34, 2832, 831, 10415, 934, 89, 4924, 4296, 549, 1238, 82, 3695, 3908, 2922, 8784, 2835, 1728, 9102, 429, 3308, 2088, 6710, 712, 868, 1018, 3879, 3464, 4503, 98, 10765, 5347, 3862, 1615, 10385, 2253, 2858, 3839, 3868, 3442, 2811, 1339, 10605, 66, 930, 149, 1007, 8488, 2691, 3494, 1903, 12331, 3732, 162, 343, 5829, 1143, 8970, 2896, 1644, 8004, 3635, 2544, 6990, 3612, 1630, 4297, 336, 3522, 4846, 2792, 1347, 390, 345, 130, 1158, 2673, 28, 11562, 3507, 1396, 398, 1324, 7743, 9888, 725, 3, 9907, 1575, 6173, 3897, 4301, 1348, 1315, 622, 6586, 447, 632, 268, 1264, 3761, 414, 449, 976, 3433, 982, 4781, 1992, 2078, 1516, 1005, 3030, 490, 6535, 1313, 642, 572, 3587, 1953, 705, 1623, 1925, 816, 1406, 2225, 10105, 7395, 1863, 1069, 489, 86, 1667, 5669, 435, 3036, 4231, 1640, 1268, 2839, 2905, 12309, 381, 1349, 272, 1169, 5488, 3360, 5173, 3252, 966, 665, 711, 1254, 2123, 1451, 3684, 520, 621, 2085, 11872, 145, 1639, 254, 1746, 543, 844, 521, 2213, 1984, 4763, 378, 5931, 764, 12343, 1385, 1569, 9105, 925, 1882, 2190, 6007, 2235, 2485, 484, 10150, 3802, 2122, 158, 1450, 175, 10, 6472, 6943, 3957, 5619, 1425, 2705, 6292, 1509, 3283, 1762, 458, 5229, 7880, 2734, 7676, 1333, 581, 1457, 3664, 3816, 669, 153, 979, 310, 526, 2486, 939, 5179, 1739, 3133, 1284, 20, 1303, 58, 2999, 876, 3219, 5143, 5063, 2821, 2304, 505, 6410, 1654, 9013, 2178, 6128, 2904, 5570, 1941, 263, 363, 12217, 420, 1311, 11737, 2231, 7726, 1847, 10407, 2054, 3595, 3396, 2685, 8050, 7699, 1133, 2361, 750, 40, 1519, 8601, 573, 2246, 535, 339, 182, 1482, 3270, 1591, 148, 305, 1750, 1015, 3047, 1096, 1600, 1928, 4424, 1538, 217, 277, 2763, 653, 944, 8265, 3015, 56, 6694, 996, 4260, 1411, 2850, 2009, 5233, 2240, 8756, 7364, 9999, 2071, 4330, 5201, 3162, 4512, 12256, 1274, 3456, 6065, 5981, 1976, 12569, 8902, 2806, 11121, 734, 3572, 5760, 1088, 27, 12545, 195, 4882, 762, 5411, 3022, 2916, 1486, 3448, 1417, 5421, 1355, 4451, 5593, 984, 767, 167, 4572, 239, 817, 875, 2711, 534, 3894, 919, 726, 6424, 999, 2755, 12246, 1656, 6577, 198, 4733, 70, 1713, 1647, 431, 4632, 569, 1636, 732, 3803, 1401, 3174, 260, 3902, 7208, 4456, 4004, 4127, 2440, 3653, 3198, 156, 1157, 8674, 5393, 1754, 5751, 670, 128, 4716, 215, 1111, 993, 3959, 2260, 1136, 4125, 2305, 4042, 1618, 4548, 2182, 426, 729, 450, 380, 491, 675, 51, 2292, 1886, 1419, 852, 3082, 2148, 359, 932, 1081, 5899, 285, 9383, 2261, 4579, 1786, 1280, 2555, 374, 2064, 2657, 3799, 324, 1521, 8353, 471, 1873, 839, 5057, 2416, 2945, 83, 424, 8851, 7742, 1779, 853, 10696, 607, 107, 2266, 12908, 12768, 3356, 8274, 5690, 4627, 1033, 2314, 686, 4019, 2940, 1949, 10597, 1172, 1091, 5036, 9460, 6419, 3278, 1204, 11061, 157, 1661, 1134, 8894, 1156, 3370, 2294, 2522, 2117, 9450, 1983, 11341, 6259, 337, 4419, 183, 3580, 3055, 7164, 81, 1272, 1503, 515, 681, 2026, 500, 1193, 2167, 5905, 988, 1567, 173, 3017, 8815, 896, 8597, 10213, 914, 4731, 351, 745, 8091, 1842, 117, 3033, 4786, 552, 1627, 234, 1161, 3048, 1101, 8084, 4105, 6224, 2503, 10029, 3290, 2146, 6722, 1023, 286, 4123, 2475, 2310, 7638, 3588, 1210, 5292, 931, 8466, 5353, 9090, 387, 2968, 781, 4751, 556, 3412, 710, 95, 1613, 143, 935, 200, 818, 680, 588, 5000, 3742, 4789, 242, 347, 1956, 2542, 204, 8958, 7267, 3683, 2623, 6935, 1194, 3554, 2409, 1809, 591, 4877, 249, 1220, 298, 1687, 1531, 1016, 9233, 3861, 7848, 3528, 2510, 2159, 1766, 245, 1367, 718, 72, 7806, 1485, 313, 104, 2915, 863, 1191, 10532, 1176, 361, 1420, 2888, 854, 728, 232, 19, 23, 3439, 2295, 4843, 1761, 8820, 2439, 11290, 2681, 923, 640, 6277, 1175, 8108, 571, 2019, 682, 1362, 1874, 738, 1642, 350, 9839, 3411, 8344, 961, 2774, 25, 3682, 5680, 132, 5134, 4907, 1579, 5661, 2684, 9045, 11903, 292, 5374, 1447, 655, 3403, 3369, 248, 2222, 488, 5318, 2239, 819, 5380, 10227, 4488, 1807, 1031, 7795, 164, 10200, 1053, 209, 1722, 3947, 755, 124, 5148, 2100, 4824, 3004, 224, 837, 228, 409, 355, 8832, 9847, 1933, 1013, 5900, 4734, 2583, 2152, 6941, 1027, 11038, 11823, 9530, 7575, 9245, 2751, 6582, 9482, 231, 2556, 2274, 464, 22, 7064, 8703, 125, 667, 5636, 2893, 2690, 2713, 587, 187, 11602, 1552, 3823, 1403, 651, 619, 525, 9521, 437, 7091, 1690, 720, 645, 4147, 1155, 473, 2519, 154, 3192, 620, 1584, 2836, 10260, 3636, 368, 0, 8514, 103, 3676, 2517, 985, 6046, 210, 11110, 11792, 2970, 4284, 1302, 73, 7186, 4187, 2716, 727, 2113, 3877, 2560, 8643, 10081, 6291, 5525, 2360, 3255, 1412, 1544, 1581, 2677, 1840, 7831, 595, 43, 289, 688, 956, 3637, 636, 4248, 2499, 133, 11974, 1909, 7604, 3623, 545, 1445, 1094, 10368, 6763, 3726, 1490, 85, 113, 2263, 739, 5396, 469, 994, 6842, 294, 10333, 391, 1583, 4544, 4341, 1165, 3933, 10502, 2228, 4386, 523, 10490, 1309, 8555, 1930, 5101, 172, 842, 1215, 4241, 6194, 754, 1054, 9725, 7506, 256, 1469, 2022, 2275, 2211, 131, 2502, 4873, 3481, 37, 3147, 386, 898, 52, 237, 492, 2801, 744, 3782, 746, 1118, 570, 2319, 1716, 1776, 1042, 7333, 6696, 2256, 722, 1386, 1835, 8818, 575, 10598, 1275, 2358, 329, 478, 6715, 1046, 1113, 1173, 1549, 1395, 3835, 5412, 7223, 4556, 5200, 4285, 6600, 7923, 2554, 2151, 9939, 11350, 3627, 388, 1369, 652, 2000, 1226, 5710, 1561, 11146, 3157, 68, 954, 3398, 6239, 321, 12482, 654, 2244, 1577, 5241, 5403, 847, 5363, 178, 662, 2669, 320, 87, 973, 470, 163, 84, 5842, 10235, 2897, 1645, 759, 5943, 1783, 3050, 1896, 3544, 382, 9369, 1352, 2396, 499, 8619, 11, 1857, 4782, 413, 9956, 789, 12023, 2724, 230, 11228, 440, 3374, 3817, 181, 2960, 8909, 1135, 2201, 10459, 366, 8845, 1900, 1276, 362, 211, 8052, 671, 9071, 5168, 12, 3844, 7, 1179, 3251, 7197, 4059, 1483, 1825, 114, 1441, 551, 7598, 1149, 6271, 5160, 824, 2049, 3715, 5276, 1562, 10602, 623, 338, 3988, 647, 6032, 191, 641, 3427, 3146, 10214, 1200, 916, 433, 4218, 582, 5220, 2311, 2290, 5272, 135, 418, 441, 3740, 1138, 4514, 2426, 1797, 3764, 793, 7440, 1506, 100, 1036, 7022, 5953, 6977, 901, 6159, 8074, 596, 2929, 584, 353, 716, 723, 7627, 4611, 1673, 2417, 7118, 222, 69, 3430, 7237, 485, 1080, 4036, 7046, 1399, 2732, 2805, 170, 1439, 2599, 509, 501, 6880, 2110, 106, 786, 459, 5866, 1282, 1964, 216, 456, 3093, 1383, 2487, 585, 10288, 1564, 290, 2534, 2255, 155, 246, 1261, 259, 580, 1250, 1452, 240, 1924, 6316, 1397, 6139, 2395, 2212, 2176, 704, 5377, 3305, 496, 3038, 4724, 2663, 1230, 280, 3319, 897, 2552, 3541, 193, 1967, 1141, 7871, 4828, 1229, 5949, 161, 6576, 1416, 518, 1336, 1970, 1775, 3969, 9159, 7083, 2333, 1151, 2577, 4972, 1056, 1116, 6138, 291, 2220, 880, 190, 7112, 4397, 1431, 567, 2483, 2323, 2558, 736, 1060, 4823, 3039, 11095, 1325, 1515, 4272, 1702, 5378, 1877, 870, 410, 111, 967, 9951, 11706, 7748, 1289, 4595, 8562, 1508, 1920, 2646, 888, 8, 6153, 574, 519, 365, 7013, 1014, 92, 221, 2590, 1671, 11413, 1649, 1512, 715, 592, 2325, 2229, 397, 1292, 943, 546, 328, 631, 892, 2918, 5073, 1770, 627, 601, 9675, 3406, 11832, 2160, 1092, 4159, 4979, 2879, 1721, 244, 5630, 139, 958, 811, 10279, 430, 5523, 1843, 7334, 5648, 465, 3745, 6143, 5255, 6898, 2463, 3624, 3765, 460, 4343, 2643, 10864, 2065, 475, 6508, 1375, 5312, 4207, 6806, 566, 4474, 1688, 2018, 3503, 4974, 6337, 1957, 602, 1643, 1370, 5904, 5528, 891, 10785, 1087, 2987, 3357, 8588, 3170, 1853, 2096, 741, 4916, 503, 1183, 4131, 2829, 1906, 12715, 3381, 683, 2055, 3202, 884, 881, 615, 6178, 295, 8308, 360, 122, 1748, 265, 2129, 4913, 7494, 2840, 357, 756, 3633, 1045, 12664, 316, 5868, 11127, 11157, 1433, 274, 1062, 8835, 964, 1652, 10504, 1572, 2664, 4164, 3602, 15, 6834, 8975, 10219, 1117, 749, 2853, 3542, 202, 14, 2637, 186, 2632, 269, 3312, 109, 480, 2551, 307, 1422, 664, 3075, 322, 577, 1034, 953, 3239, 9573, 10494, 1462, 1505, 1105, 1263, 71, 1465, 508, 719, 6258, 1774, 4071, 1714, 672, 1279, 3267, 4768, 443, 9003, 4371, 6163, 8725, 419, 1706, 76, 273, 1026, 53, 214, 4629, 1247, 9739, 768, 3086, 8448, 1114, 5199, 4454, 4246, 3006, 890, 4649, 3838, 9961, 1082, 8999, 1232, 1028, 3537, 1322, 1602, 50, 594, 2671, 706, 1456, 1235, 3974, 1343, 205, 9308, 4783, 5310, 679, 1697, 270, 867, 5465, 4690, 6087, 3648, 8226, 886, 438, 5286, 118, 67, 3672, 2793, 1241, 3014, 495, 4049, 1190, 4028, 634, 3127, 2379, 16, 1186, 1, 5650, 31, 8388, 146, 7759, 1829, 225, 2919, 2002, 6900, 2676, 3719, 659, 7904, 3691, 883, 5731, 1975, 3345, 915, 1390, 6531, 88, 428, 4784, 2, 102, 3199, 1213, 1112, 5275, 2380, 4818, 2585, 1610, 3788, 2706, 579, 340, 1472, 127, 188, 7594, 1223, 8402, 1043, 472, 5316, 3531, 1102, 5893, 5079, 1619, 5519, 1912, 406, 3090, 5040, 1669, 12413, 369, 8458, 2061, 385, 5881, 6111, 612, 613, 1332, 2158, 5408, 3409, 1262, 6870, 6692, 4665, 3929, 110, 147, 306, 5464, 2016, 3175, 3923, 6746, 1171, 5056, 6066, 791])
    test_idx_map = np.array([1010, 1343, 98, 313, 27, 338, 946, 539, 235, 447, 1203, 1076, 2232, 10, 638, 192, 1310, 284, 74, 215, 169, 281, 111, 675, 531, 359, 1026, 1103, 560, 352, 276, 52, 822, 287, 527, 526, 1434, 180, 44, 422, 90, 378, 206, 97, 315, 1714, 154, 3, 179, 814, 738, 869, 122, 440, 161, 490, 231, 1047, 1378, 659, 134, 131, 1006, 9, 462, 246, 2142, 390, 37, 895, 204, 369, 836, 781, 759, 1414, 1176, 1238, 325, 493, 6, 82, 1288, 105, 114, 594, 1435, 887, 81, 66, 796, 540, 1234, 286, 736, 1621, 201, 104, 283, 1716, 1768, 138, 1099, 384, 1505, 548, 270, 149, 387, 630, 396, 1825, 70, 603, 559, 67, 25, 195, 1251, 123, 731, 2210, 1581, 160, 877, 2078, 249, 1150, 129, 692, 388, 450, 65, 1297, 307, 691, 75, 374, 357, 430, 342, 655, 35, 2132, 748, 182, 156, 701, 291, 393, 1163, 866, 515, 1074, 279, 558, 399, 61, 68, 367, 63, 0, 277, 434, 597, 882, 356, 106, 664, 599, 986, 1830, 273, 184, 607, 870, 608, 1662, 336, 397, 432, 91, 1053, 121, 77, 46, 320, 302, 2233, 294, 176, 646, 187, 53, 143, 1612, 103, 792, 669, 2405, 26, 223, 1233, 500, 227, 661, 295, 40, 737, 159, 167, 188, 580, 983, 1765, 479, 455, 1049, 36, 1622, 117, 110, 1167, 328, 314, 267, 32, 583, 1628, 606, 22, 463, 466, 930, 2195, 628, 1003, 130, 802, 1710, 505, 647, 58, 1345, 405, 370, 1085, 1128, 705, 1523, 488, 533, 128, 421, 155, 31, 1138, 1000, 12, 525, 2151, 212, 549, 16, 101, 207, 2187, 343, 275, 631, 178, 327, 42, 514, 158, 141, 1283, 1385, 1548, 361, 319, 39, 148, 85, 389, 1389, 910, 1084, 33, 457, 601, 460, 768, 2573, 551, 1812, 168, 899, 24, 1863, 392, 653, 300, 272, 413, 1017, 41, 1058, 107, 1572, 574, 62, 1080, 49, 5, 112, 146, 59, 476, 761, 71, 681, 1418, 687, 1670, 2237, 304, 728, 163, 4, 301, 542, 142, 29, 19, 30, 570, 296, 2072, 536, 96, 1312, 7, 102, 893, 501, 43, 1032, 1449, 229, 1073, 693, 83, 402, 119, 198, 268, 288, 1402, 567, 20, 475, 2325, 17, 856, 48, 541, 371, 316, 713, 137, 355, 732, 177, 643, 800, 523, 782, 1754, 419, 339, 57, 2555, 99, 109, 13, 521, 620, 56, 784, 794, 445, 375, 69, 1828, 60, 145, 803, 1585, 377, 939, 1075, 708, 2293, 358, 908, 92, 233, 408, 164, 79, 926, 54, 72, 1632, 15, 1805, 132, 1281, 144, 1142, 191, 1243, 218, 1447, 1983, 88, 120, 242, 711, 108, 80, 613, 18, 51, 1059, 806, 185, 173, 312, 55, 1264, 73, 274, 1259, 519, 348, 2, 126, 1, 349, 1424, 28, 217, 34, 666, 136, 381, 76, 157, 568, 787, 1221, 2059, 14, 733, 209, 171, 1639, 89, 47, 175, 651, 1008, 744, 2282, 252, 443, 485, 654, 1282, 394, 415, 1853, 11, 331, 139, 38, 364, 922, 1834, 969, 554, 170, 923, 23, 133, 577, 411, 151])

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        # For this implementation, we do a 2-level index lookup. For every incremental index
        # in this distilled dataset, we do a lookup into the full dataset location.
        idx_map = self.train_idx_map if self.train else self.test_idx_map
        full_idx = idx_map[index]
        return super().__getitem__(full_idx)

    def __len__(self) -> int:
        idx_map = self.train_idx_map if self.train else self.test_idx_map
        return len(idx_map)
        

class CIFAR10_Trainer(L.LightningModule):
    def __init__(self, classifier_model: nn.Module, model_name: str, batchsz: int, use_qat: bool = True, export_on_end: bool = False, lr: float = 1e-3):
        super().__init__()
        self.model_name = model_name
        self.classifier_model = classifier_model
        self.batchsz = batchsz
        self.lr = lr
        self.loss_fn = CrossEntropyLoss()
        self.prev_epoch_step = 0
        self.val_accuracy = 0
        self.val_batch_count = 0
        self.use_qat = use_qat
        self.export_on_end = export_on_end
        self.dump_fx_graphs = False
        # the batch dimension needs to be > 1, because it will fail some batchnorm checks in the exporter if not.
        self.dummy_inputs = (torch.randn(self.batchsz, 3, 32, 32), )
        # Call this last once all init has been done
        self.save_hyperparameters(ignore=['classifier_model'])

    def _get_onnx_name(self) -> str:
        return f"{self.model_name}_model.onnx"

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.lr)
        # We use the simplest linear LR schedule decay. These are unit tests which run for no more than
        # a few epochs.
        scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.5, total_iters=4)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1,
            },
        }

    def forward(self, imgs):
        # Forward function that is run when visualizing the graph
        return self.classifier_model(imgs)
        
    def _step(self, batch, batch_idx):
        x, gt = batch
        logits_y = self.classifier_model(x)
        loss = self.loss_fn(logits_y, gt)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch, batch_idx)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, gt = batch
        logits_y = self.classifier_model(x)
        loss = self.loss_fn(logits_y, gt)
        # We run the validation set here.
        self.log("val_loss", loss, prog_bar=True)
        # Compute top-1 accuracy here and keep a running tab of results.
        scores = torch.argmax(logits_y, dim=-1) == gt
        self.val_accuracy += torch.mean(scores.type(torch.float32))
        self.val_batch_count += 1
        top1_acc = self.val_accuracy / self.val_batch_count
        self.log("val_acc", top1_acc, prog_bar=True)
        return loss
    
    def on_validation_end(self) -> None:
        super().on_validation_end()
        top1_acc = self.val_accuracy / self.val_batch_count
        # self.log("val_acc", top1_acc)
        self.val_accuracy = 0
        self.val_batch_count = 0
        self.prev_epoch_step = self.global_step 
        if self.global_step > 0:
            print(f"Validation top-1 accuracy, epoch {self.current_epoch}: {top1_acc}")

    def on_train_start(self) -> None:
        super().on_train_start()
        if self.use_qat:
            self._prepare_qat()
        else:
            # Do a compile so we can see an FX graph
            # print(f"Compiling model to FX graph ...")
            # m = torch.compile(self.mnist_model)
            # setattr(self, 'mnist_model', m)
            self._dump_fx_graph('compiled_graph.txt')
        pass

    def on_train_end(self) -> None:
        super().on_train_end()
        self._finalize_qat_model()

    def on_train_epoch_start(self) -> None:
        # For some reason Lightning doesn't switch to train mode ???
        self.train(True)
        lr = self.trainer.lr_scheduler_configs[0].scheduler.get_last_lr()[0]
        self.log('learning_rate', lr, on_step=False, on_epoch=True, prog_bar=True)

    def _prepare_qat(self) -> None:
        m = sima_prepare_qat_model(input_graph=self.classifier_model, inputs=self.dummy_inputs, device=self.device)
        # Now replace our model
        setattr(self, 'classifier_model', m)
        self._dump_fx_graph('prepare_fx_qat_graph.txt')

    def _finalize_qat_model(self) -> None:
        self.train(False)
        # If we are running in QAT mode, convert to a quantized graph.
        if self.use_qat:
            m = sima_finalize_qat_model(self.classifier_model)
            # Now replace our model
            setattr(self, 'classifier_model', m)
            self._dump_fx_graph('final_fx_qat_graph.txt')
        return

    def on_fit_end(self) -> None:
        if self.export_on_end:
            self.to_onnx(file_path=self._get_onnx_name())

    def _dump_fx_graph(self, fname: str) -> None:
        if not self.dump_fx_graphs:
            return
        if not isinstance(self.classifier_model, GraphModule):
            from torch.fx import symbolic_trace
            # Symbolic tracing frontend - captures the semantics of the module
            symbolic_traced : torch.fx.GraphModule = symbolic_trace(self.classifier_model)
            # If the graph is not already in FX format, we skip this entire step.
            # return
        else:
            symbolic_traced = self.classifier_model

        # High-level intermediate representation (IR) - Graph representation
        print(f"Generating graph dump to file: {fname}")
        with open(fname, 'wt') as f:
            print(symbolic_traced.graph, file=f)

    def to_onnx(self, file_path: str | Path, input_sample: Any | None = None, **kwargs: Any) -> None:
        """ This function needs to be overridden in the case of QAT, since export gets tricky
            and specialized.
        """
        self.train(False)
        print(f"Writing onnx file output to: {file_path}")
        sima_export_onnx(qat_model=self.classifier_model, inputs=self.dummy_inputs, output_file=file_path)
    
    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """ We have to apply the QAT scaffold before we load a checkpoint (if QAT is enabled),
            because Pytorch doesn't serialize the graph, just the params. Pytorch changes
            all the param names when scaffolding is applied, so the state_dict will have 
            mismatching keys unless we scaffold the model first.
        """
        if self.use_qat:
            self._prepare_qat()
        return super().on_load_checkpoint(checkpoint)


def build_dataloaders(args: Namespace):
    """ Build the dataset iterators
    """
    cifar10_normalization = transforms.Normalize(
        mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
        std=[x / 255.0 for x in [63.0, 62.1, 66.7]],
    )

    train_transforms = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            cifar10_normalization,
        ]
    )
    test_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            cifar10_normalization,
        ]
    )

    dataset_train = CIFAR10Mini(args.data, train=True, download=True, transform=train_transforms)
    dataset_test = CIFAR10Mini(args.data, train=False, download=True, transform=test_transforms)

    workers = 0
    train_dataloader = DataLoader(dataset_train, batch_size=args.batch, shuffle=True, num_workers=workers) # persistent_workers=True)
    test_dataloader = DataLoader(dataset_test, batch_size=args.batch, shuffle=False, num_workers=workers) # , persistent_workers=True)
    return train_dataloader, test_dataloader


def run_train(args: Namespace, cifar10_trainer: L.LightningModule):
    """ Run the CIFAR10 training regimen.
    """
    cifar10_trainer.to(args.device)

    train_dataloader, test_dataloader = build_dataloaders(args)

    pytest_mode = os.environ.get("PYTEST_VERSION") is not None
    print(f"Pytest mode is: {pytest_mode}")

    trainer = L.Trainer(
        enable_checkpointing=False,     # Needed to prevent problems with torch.fx
        logger=False,                   # Regression tests will generate too many logs
        max_epochs=args.epochs, 
        accelerator=args.device, 
        default_root_dir='.',
        enable_progress_bar=(not pytest_mode),
    )
    # We will use the test set as our validation set. It just needs to be out-of-sample,
    # and it eliminates the need for dataset splits.
    trainer.fit(
        model=cifar10_trainer, 
        train_dataloaders=train_dataloader,
        val_dataloaders=test_dataloader,
    )
    # Make sure inference mode succeeds
    try:
        cifar10_trainer.eval()
    except:
        assert False, "Model trainer failed on call to eval() after training"
    assert cifar10_trainer.classifier_model.training == False, "Model trainer failed to set evaluation mode after training"

    # Here we make sure that putting the graph in training mode will fail.
    rc = False
    try:
        cifar10_trainer.train()
    except:
        # This is expected
        rc = True
        pass
    assert rc, "Failed to catch exception in fq mode when train(True)"
    assert cifar10_trainer.classifier_model.training == False

    # Do a pass on the test set after training. This will exercise the Q/DQ version of the graph. We use the
    # validate() version in order to re-use the metrics created for the training monitor.
    lit_val_metrics = trainer.validate(
        model=cifar10_trainer,
        dataloaders=test_dataloader,
    )
    print(f"Training Done!")
    # After training, we will return the achieved accuracy on the validation set.
    # We need to flatten all the metrics into one dictionary, because the list format used by Lightning
    # is awkward and hard to access.
    val_metrics = reduce(or_, lit_val_metrics, {})
    return val_metrics


def get_ort_session(onnx_file: str) -> onnxruntime.InferenceSession:
    # We need to deliberately keep the onnx-runtime down to a very small footprint. This is crucial
    # for regressions, since we will have many tests running in parallel.
    sess_options = onnxruntime.SessionOptions()
    sess_options.intra_op_num_threads = 1
    sess_options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
    sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.add_session_config_entry("session.intra_op.allow_spinning", "0")
    return onnxruntime.InferenceSession(onnx_file, sess_options=sess_options)


def onnxrt_test(args: Namespace, onnx_file: str) -> float:
    """ Run a trained model in onnx-runtime and verify the accuracy. This is independent 
        verification that a model is exported correctly and meets expected accuracy goals.
    """
    ort_session = get_ort_session(onnx_file)

    # get the name of the first input of the model
    input_t = ort_session.get_inputs()[0]
    onnx_batch_sz = int(input_t.shape[0])
    # Use the Lightning dataloaders to iterate over the dataset
    _, test_dataloader = build_dataloaders(args)

    test_accuracy = 0.
    test_batch_count = 0
    print(f"Running ONNX model: {onnx_file}")
    for s in tqdm(test_dataloader):        
        # list: [sample, gt]
        samples_batch = np.array(s[0])
        test_batch_sz = int(samples_batch.shape[0])
        # If we are on the last batch and it doesn't match the ONNX batch size, pad it up
        if onnx_batch_sz != test_batch_sz:
            samples_batch = np.pad(samples_batch, pad_width=((0, onnx_batch_sz-test_batch_sz), (0,0), (0,0), (0,0)))
        ort_inputs = {input_t.name: samples_batch}
        outs = ort_session.run(None, ort_inputs)
        outs = outs[0][:test_batch_sz]
        classes = np.argmax(outs, axis=1)
        scores = (classes == np.array(s[1]))
        test_accuracy += np.mean(scores.astype(np.float32))
        test_batch_count += 1

    top1_acc = test_accuracy / test_batch_count
    print(f"Got ONNX accuracy: {top1_acc}")
    return top1_acc


def training_test(args: Namespace, classifier_model: nn.Module, model_name: str) -> bool:
    """ Run a standard training style test against the model.
    """
    # Set the global seed to prevent headaches.
    L.seed_everything(42)
    min_acc = args.acc

    trainer = CIFAR10_Trainer(
        model_name=model_name,
        classifier_model=classifier_model,
        export_on_end=True, 
        use_qat=(not args.disable_qat),
        batchsz=args.batch,
        lr=args.lr,
    )

    metrics = run_train(args, trainer)
    # metrics = {'val_loss': 1.5993834733963013, 'val_acc': 0.3838767409324646}
    # print(f"Got metrics: {metrics}")
    pytorch_val_acc = metrics['val_acc']
    if pytorch_val_acc < min_acc:
        print(f"Validation accuracy {pytorch_val_acc} below minimum threshold: {min_acc}")
        return False
    # Now invoke onnx-runtime
    ort_acc = onnxrt_test(args, trainer._get_onnx_name())
    if ort_acc < min_acc:
        print(f"ONNX runtime accuracy {ort_acc} below minimum threshold: {min_acc}")
        return False
    return True


def get_args():
    parser = argparse.ArgumentParser(description=f"Train CIFAR10 using QAT")
    parser.add_argument('-e', '--epochs', type=int, default=2, help='Epochs to train')
    parser.add_argument('-b', '--batch', type=int, default=16, help='Batch size')
    parser.add_argument('-l', '--lr', type=float, default=1e-3, help='Initial learning rate')
    parser.add_argument('-a', '--acc', type=float, help='Expected accuracy threshold')
    parser.add_argument('-d', '--data', type=str, default="./data", help='Dataset location')
    parser.add_argument('--device', type=str, default="cpu", help='Device to use')
    parser.add_argument('--disable-qat', action='store_true', help='Disable QAT mode')
    all_args = parser.parse_args()
    return all_args

