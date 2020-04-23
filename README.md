# Tacotron Korean TTS

[Keith Ito](https://github.com/keithito)의 타코트론 코드로 한국어 TTS를 구현합니다

based on
  * https://github.com/keithito/tacotron

logs-tacotron 폴더에 135000까지 학습한 모델이 있습니다

### Training

1. **한국어 음성 데이터 다운로드**

    * [KSS](https://www.kaggle.com/bryanpark/korean-single-speaker-speech-dataset)

2. **`~/Tacotron-Korean`에 학습 데이터 준비**

   ```
   Tacotron-Korean
     |- kss
         |- 1
         |- 2
         |- 3
         |- 4
         |- transcript.v.1.x.txt
   ```

3. **Preprocess**
   ```
   python preprocess.py
   ```
     * 실행 전에 preprocess.py를 열어서 transcript 버전이 일치하는지 확인해야 합니다

4. **Train**
   ```
   python train.py
   ```

   재학습 시
   ```
   python train.py --restore_step 135000
   ```
     * 숫자를 변경하면 됩니다

5. **Synthesize**
   ```
   python eval.py --checkpoint ~/tacotron/logs-tacotron/model.ckpt-135000
   ```
     * eval.py를 열어서 합성할 문장을 정하면 됩니다

#### Attention alignment example
![ezgif com-gif-maker](https://user-images.githubusercontent.com/49984198/77516623-13029780-6ebe-11ea-8bd2-821930682ea8.gif)


윈도우에서 Tacotron 한국어 TTS 학습하기
  * https://chldkato.tistory.com/141
  
Tacotron 정리
  * https://chldkato.tistory.com/143
