import numpy as np
import csv
import os
import time
import joblib
from scipy.optimize import curve_fit
from Bar_inspection_func_v2 import *

# 결함진단 관련 파라미터 정의
L_shortRange = 20
X_clst_ref = 10
thresh_V_def = 2.0
thresh_V_delete = 1.5
thresh_V_BR1 = 8.0
thresh_V_BR2 = 5.5
thresh_X_diff_BR2_1 = 4
thresh_X_diff_BR2_2 = 15
range_V_invalid = 0.1
N_window = 10

# 입/출력 경로(폴더) 지정
DATA_dir = ['../NO1_DATA', '../NO2_DATA']
RESULT_dir = ['../NO1_RESULT', '../NO2_RESULT']

while (1) :
    if (len(DATA_dir) != len(RESULT_dir)):
        print("The number of input(DATA) and output(RESULT) folders must be the same.")
        exit()

    # input/output 폴더 개수만큼 반복
    for fileDir_in, fileDir_out in zip(DATA_dir, RESULT_dir):
        fileList = os.listdir(fileDir_in)

        # 폴더가 비어있는 경우 다음단계로 건너뜀
        if (len(fileList) == 0):
            print('[ Waiting for MLFT files (path: {0}) ]\n'.format(fileDir_in))
            time.sleep(1)
            continue

        # 폴더에 파일 존재시 4초간 대기(접근충돌문제 회피를 위함)
        else:
            print('[{0} files detected in ''{1}'']\n'.format(str(len(fileList)), fileDir_in))
            time.sleep(4)

        # 폴더 내 DATA 개수만큼 반복
        for i_file, fileName in enumerate(fileList):

            filePath = fileDir_in + '/' + fileName

            print('DATA :\t', filePath)
            if (invalidDataCheck(filePath)):
                print('WARNING! invalid data\n')
                if os.path.isfile(filePath):
                    os.remove(filePath)
                continue

            # MLFT 데이터 정보 추출
            DATE, HNO, TOTAL, V = getDataInfo(filePath)
            L = len(V)
            X = np.arange(0, L)
            Class = np.zeros(L)

            if (max(V) < thresh_V_def):
                print('WARNING! ', max(V), '(V_max) < 2.0 : Exception)\n')
                if os.path.isfile(filePath):
                    os.remove(filePath)
                continue
        
            # 로컬(세부구간)데이터 추출
            X_local, V_local = getLocalData(V, thresh_V_def, L_shortRange)
            N_local = len(V_local)

            V = np.array(V)
            X_local, V_local = np.array(X_local), np.array(V_local)
        
            # 세부구간 결함특성 구분
            class_local = np.empty((N_local, L_shortRange))
            clf_from_joblib = joblib.load('SVM_model_v2.pkl') 
            for i in range(N_local):
                V_local_temp = V_local[i,:]
                V_local_temp = np.where(V_local_temp >= thresh_V_delete, V_local_temp, 0)
                V_local_temp = np.where(V_local_temp >= thresh_V_def, thresh_V_def, V_local_temp)
                pred = clf_from_joblib.predict([V_local_temp])
                class_local[i,:L_shortRange] = pred[0]
                
            # 세부구간 결함정보가 커플링된 경우 보정
            X_prev = X_local[0,:]
            for i in range(1, N_local):
                X_curr = X_local[i,:]
                X_diff = int(X_curr[0] - X_prev[0])
                if (X_diff < X_clst_ref):
                    temp_max_class = np.amax([class_local[i-1,X_diff:], class_local[i,:L_shortRange-X_diff]])
                    class_local[i-1,:] = temp_max_class
                    class_local[i,:] = temp_max_class
                X_prev = X_curr

            X_next = X_local[N_local-1,:]
            for i in range(N_local-2, 0, -1):
                X_curr = X_local[i,:]
                X_diff = int(X_next[0] - X_curr[0])
                # if (X_diff < L_shortRange):
                if (X_diff < X_clst_ref):
                    temp_max_class = np.amax([class_local[i,X_diff:], class_local[i+1,:L_shortRange-X_diff]])
                    class_local[i+1,:] = temp_max_class
                    class_local[i,:] = temp_max_class
                X_next = X_curr

            # 세부구간 결함정보 통합
            X_merge = np.zeros(L)
            V_merge = np.zeros(L)
            class_merge = np.zeros(L)
            for i in range(0, N_local):
                X_merge[X_local[i,:]] = X_local[i,:]
                V_merge[X_local[i,:]] = V_local[i,:]
                class_merge[X_local[i,:]] = class_local[i,:]

            X_merge = X_merge[class_merge!=0]
            V_merge = V_merge[class_merge!=0]
            class_merge = class_merge[class_merge!=0]

            # 초기 클러스터 정보 생성
            L_merge = len(X_merge)
            N_clst = 1
            clst_merge = np.zeros(L_merge)
            clst_merge[0] = N_clst
            for i in range(1, L_merge):
                if (X_merge[i] - X_merge[i-1] != 1 or class_merge[i] != class_merge[i-1]):
                    N_clst += 1
                clst_merge[i] = N_clst
            
            # 결함 발생구간 결정
            X_def = np.array([])
            V_def = np.array([])
            class_def = np.array([]) 
            clst_def = np.array([]) 

            idx_clst = 1
            for i in range(1,N_clst+1):  
                roi_idx = np.where(clst_merge == i)
                X_roi = X_merge[roi_idx]
                V_roi = V_merge[roi_idx]
                class_roi = class_merge[roi_idx]

                if (class_roi[0] == 1):
                    remain_idx = np.where(V_roi >= thresh_V_def)
                    remain_idx = remain_idx[0]   
                elif (class_roi[0] == 2):
                    remain_idx = np.where(V_roi >= thresh_V_delete)
                    remain_idx = remain_idx[0]   
                    remain_idx = np.arange(remain_idx[0], remain_idx[-1]+1)

                X_def = np.append(X_def, X_roi[remain_idx])
                V_def = np.append(V_def, V_roi[remain_idx])
                class_def = np.append(class_def, class_roi[remain_idx])
            X_def = X_def.astype(int)

            # 클러스터 정보 생성
            L_def = len(X_def)
            N_clst = 1
            clst_def = np.zeros(L_def)
            clst_def[0] = N_clst
            for i in range(1, L_def):
                if (X_def[i] - X_def[i-1] != 1 or class_def[i] != class_def[i-1]):
                    N_clst += 1
                clst_def[i] = N_clst
            
            # 결함별 특성에 초기코드 부여
            for i in range(1,N_clst+1):
                roi_idx = np.where(clst_def == i)
                roi_idx = roi_idx[0]
                V_roi = V_def[roi_idx]
                if (class_def[roi_idx[0]] <= 2 and np.max(V_roi) >= thresh_V_BR1):
                    class_def[roi_idx] = (-class_def[roi_idx] + 7) * 0.5    # DO NOT CHANGE
                    if (class_def[roi_idx[0]] == 3 and len(roi_idx) >= 3):
                        class_def[roi_idx] = 1

            # 통계적 특성 산출
            V_sort = np.sort(V)
            vaildIdx = range(int(L*range_V_invalid),int(L*(1-range_V_invalid)))
            V_sort_cut = V_sort[vaildIdx]

            xdata = range(1,len(V_sort_cut)+1)
            ydata = V_sort_cut
            popt, pcov = curve_fit(funcEQ, xdata, ydata, maxfev=10000)

            fitCoeff_cut = -popt[1]
            V_mean_cut = np.mean(V_sort_cut)
            V_std_cut = np.std(V_sort_cut)

            # noise 찾기
            V_noise_maf = np.min(V)
            V_min = np.min(V)
            
            V_maf = np.zeros(L-N_window+1)
            for i in range(L-N_window+1):
                V_maf[i] = np.mean(V[i:i+N_window-1])
            V_noise_maf = np.min(V_maf)

            V_d2 = np.zeros(L)
            for i in range(2,L-2):
                V_d2[i] = (-V_sort[i-2]+16*V_sort[i-1]-30*V_sort[i]+16*V_sort[i+1]-V_sort[i+2]) / 12
            
            idx_th_ns = (np.abs(V_d2) <= 0.02)
            idx_th_ns = idx_th_ns.astype(int)
            idx_th_ns = np.append(idx_th_ns, [0, 0])

            Vns_beginIdx = 0
            Vns_len = 0
            Vns_beginIdx_curr = 0
            Vns_len_curr = 1

            bCont = 0
            if (idx_th_ns[0] == 1):
                bCont = 1

            for i in range(len(idx_th_ns)-2):
                if (bCont):
                    if (idx_th_ns[i+1] + idx_th_ns[i+2] != 0):
                        Vns_len_curr += 1
                        if (Vns_len_curr > Vns_len):
                            Vns_beginIdx = Vns_beginIdx_curr
                            Vns_len = Vns_len_curr
                    else:
                        bCont = 0
                else:
                    if (idx_th_ns[i] == 1):
                        Vns_beginIdx_curr = i
                        Vns_len_curr = 1
                        bCont = 1

            V_ns_idx = range(Vns_beginIdx,Vns_beginIdx+Vns_len)
            V_ns_median = V_sort[int(np.mean(V_ns_idx))]
            V_ns_mean = np.mean(V_sort[V_ns_idx])
            V_ns_std = np.std(V_sort[V_ns_idx])

            # class_def를 MLFT 길이단위로 병합
            Class[X_def] = class_def

            # 접힘판정 (선단/말단)
            bBR2 = 0
            for i in range(1,N_clst+1):
                roi_idx = np.where(clst_def == i)
                roi_idx = roi_idx[0]
                X_roi = X_def[roi_idx]
                V_roi = V_def[roi_idx]
                class_roi = class_def[roi_idx]

                if (class_roi[0] == 2.5):

                    idx_temp = roi_idx
                    idx_temp_num = np.sum(class_roi == 2.5)

                    if (idx_temp_num):
                        idxA = idx_temp[0]
                        idxB = idx_temp[-1]

                        A_len = np.sum(np.array(range(X_def[idxA],thresh_X_diff_BR2_1)))
                        B_len = np.sum(np.array(range(L-thresh_X_diff_BR2_1, X_def[idxB]+1)))
                        A_max1 = -1
                        B_max1 = -1
                        if (A_len > 0 and X_def[idxA] <= thresh_X_diff_BR2_1):
                            A_max1 = np.max(V[range(X_def[idxA],thresh_X_diff_BR2_1)])
                            A_max2 = np.max(V[range(X_def[idxA],thresh_X_diff_BR2_2)])

                        if (B_len > 0 and X_def[idxB] >= L-thresh_X_diff_BR2_1):
                            B_max1 = np.max(V[range(L-thresh_X_diff_BR2_1, X_def[idxB]+1)])
                            B_max2 = np.max(V[range(L-thresh_X_diff_BR2_2, X_def[idxB]+1)])

                        if ((X_def[idxA] < thresh_X_diff_BR2_1 and A_max1 > thresh_V_BR2 and A_max2 > thresh_V_BR1) or (X_def[idxB] >= L-thresh_X_diff_BR2_1 and B_max1 > thresh_V_BR2 and B_max2 > thresh_V_BR1)):
                            idx_BR2 = np.arange(X_def[idxA],X_def[idxB]+1)
                            Class[idx_BR2] = 4
                            bBR2 = 1

            # 주름판정
            bBR3 = 0
            L_BR2 = np.where(Class == 4)
            if (fitCoeff_cut >= 0.0013 and fitCoeff_cut <= 0.004 and V_mean_cut >= 1.0 and V_mean_cut <= 2.8 and V_std_cut >= 0.13 and V_std_cut <= 0.7 and len(L_BR2[0]) == 0):
                idx_BR1 = np.where((Class == 2.5) & (V >= thresh_V_BR1))
                Class[idx_BR1] = 3
                idx_BR3 = np.where(Class < 3)
                Class[idx_BR3] = 5
                bBR3 = 1

            # 접힘판정 관련 특징점 수집
            peak_dist = -1
            peak_dist2 = -1
            peak_x_diff = -1
            peak_x_diff_mean = -1
            peak_x_diff_mean_weighted = -1
            peak_x_diff_std = -1
            peak_x_diff_std_weighted = -1
            peak_x_mean = -1
            peak_x_std = -1

            noise15_num = np.sum(V >= 1.5)
            noise30_num = np.sum(V >= 3.0)
            noise60_num = np.sum(V >= 6.0)
            noise80_num = np.sum(V >= 8.0)
            noise15_rate = noise15_num / L
            noise30_rate = noise30_num / L
            noise60_rate = noise60_num / L
            noise80_rate = noise80_num / L
            peak_num = np.sum(V >= 6.0)
            peak_dist_mean = 0
            if (max(V) > 8.0 and peak_num > 1):
                peak_x = np.where(V >= 6.0)
                peak_x = peak_x[0]
                peak_V = V[peak_x]
                peak_dist = (peak_x[-1] - peak_x[0]) / L
                peak_x_diff = peak_x[1:] - peak_x[:-1]
                peak_x_diff_mean = np.mean(peak_x_diff)
                peak_x_diff_mean_weighted = peak_x_diff_mean * len(peak_x_diff)
                peak_x_diff_std = np.std(peak_x_diff)
                peak_x_diff_std_weighted = peak_x_diff_std * len(peak_x_diff)
                peak_x_mean = np.mean(peak_x)
                peak_x_std = np.std(peak_x)
                peak_x2 = np.where(V >= 5.0)
                peak_dist2 = (peak_x2[-1] - peak_x2[0]) / L

            # 전장성 결함 판정
            if (bBR3 == 1 and ((V_ns_std > 0.1 and peak_dist > 0.6 and peak_x_diff_mean_weighted > 480 and peak_x_diff_std_weighted < 1000) or (peak_dist > 0.4))):
                idx_BR2 = np.where(Class < 6)
                Class[idx_BR2] = 4
                bBR2 = 1

            # AR2 vs BR2
            L_AR2 = np.sum(Class == 2) + np.sum(Class == 2.5)
            L_BR2 = np.sum(Class == 4)
            if (L_AR2 > L_BR2):
                Class[np.where(Class == 4)] = 2
                bBR2 = 0
            else:
                Class[np.where(Class == 2)] = 4
                Class[np.where(Class == 2.5)] = 4
                bBR2 = 1

            # BR2 vs BR3
            L_BR2 = np.sum(Class == 4)
            if (L_BR2 > 0):
                Class[np.where(Class == 5)] = 4
                bBR2 = 1

            # AR2 추가 판정
            std_mean_rate = peak_x_diff_std_weighted / peak_x_diff_mean_weighted
            dist_mean_rate = peak_x_diff_mean_weighted / (peak_dist * 100)
            if (bBR2 and (std_mean_rate > 1.86 or V_ns_median > 9)):
                idx_AR2 = np.where(Class == 4)
                Class[idx_AR2] = 2
                bBR2 = 0

            # voltage noise 클경우 접힘으로 판정
            if ((V_ns_median > 3.0 and V_ns_median < 6)):
                idx_BR2 = np.where(Class < 6)
                Class[idx_BR2] = 4
                bBR2 = 1

            # 코드 출력
            Class = Class.astype(int)
            L_def = np.sum(Class > 0)
            L_AR1 = np.sum(Class == 1)
            L_AR2 = np.sum(Class == 2)
            L_BR1 = np.sum(Class == 3)
            L_BR2 = np.sum(Class == 4)
            L_BR3 = np.sum(Class == 5)

            def_name = np.array(['AR1', 'AR2', 'BR1', 'BR2', 'BR3'])
            def_ratio = np.array([L_AR1, L_AR2, L_BR1, L_BR2, L_BR3]) * 100 / L_def
            def_name = def_name[np.where(def_ratio > 0)]
            def_ratio = def_ratio[np.where(def_ratio > 0)]

            def_order = np.flip(np.argsort(def_ratio))
            def_ratio = def_ratio.astype(int) % 100

            DCODE = ''
            temp_sum = 0
            for i in range(0, 3):
                if (i < len(def_ratio) - 1):
                    temp_sum += def_ratio[def_order[i]]
                    DCODE += (def_name[def_order[i]] + str(def_ratio[def_order[i]]).zfill(2))
                elif (i == len(def_ratio) - 1):
                    temp_val = (100 - temp_sum) % 100
                    DCODE += (def_name[def_order[i]] + str(temp_val).zfill(2))
                else:
                    DCODE += 'S0000'
            print('DCODE:\t', DCODE, '\n')

            # 결과 파일 저장
            RESULT = [['HNO', HNO], ['TOTAL', str(TOTAL)], ['DATE', DATE], ['DCODE', DCODE], ['DLEN', str(L_def * 13)]]
           
            f_write = open(fileDir_out + '/' + fileName[:-4] + '_output.csv', 'w', newline='')
            writer = csv.writer(f_write)
            writer.writerows(RESULT)
            f_write.close()

            # 결함정보 출력된 MLFT 데이터 삭제
            if os.path.isfile(filePath):
                os.remove(filePath)