# danmer_cv

### 04/23
_dtw algorithm 사용하여 두 좌표 비교예정 아직 시도중..!/중간보고서 제출할때 NN도 정해서 보고할 예정임  
_dtw package는 사용해봄 정규화하면 나름 나쁘지 않은듯  
_https://github.com/Maghoumi/pytorch-softdtw-cuda#soft-dtw-for-pytorch-in-cuda 사용해서 비교해보고 더 좋은 결과를 쓸 예정  
-스크린샷 결과기준 tensor 1 tutor 2 tutee 3 가까이서 찍은 heymama 4 비슷한 거리에서 와이드하게 찍은 heymama  
3은 거의 다른 영상 수준으로 인식. 비슷한거리에서 찍으면 엄청 비슷하게 인식함  
어떻게 자를건지 지금 짧은 프레임 기준으로 임의로 자른것임. moving window 기법쓸지 NN수정할지  
dtw package에서와 cuda에서의 같은 경향성 보임 normalize의 차이만 존재함 따라서 프레임 임의로 잘라도 딱히 상관없을듯 아니 근데 프레임별로 오차를 보이려면....?


