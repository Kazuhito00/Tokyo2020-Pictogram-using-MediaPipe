# Tokyo2020-Pictogram-using-MediaPipe
MediaPipeで姿勢推定を行い、Tokyo2020オリンピック風のピクトグラムを表示するデモです。

https://user-images.githubusercontent.com/37477845/127340964-5378706f-034a-4920-be23-c6fbca442686.mp4

# Requirement 
* mediapipe 0.8.6 or later
* OpenCV 3.4.2 or later

#  Demo
以下コマンドでデモを起動してください。<br>
ESCキー押下でプログラム終了します。<br>
```
python main.py
```
* --device<br>
カメラデバイス番号の指定<br>
デフォルト：0
* --width<br>
カメラキャプチャ時の横幅<br>
デフォルト：640
* --height<br>
カメラキャプチャ時の縦幅<br>
デフォルト：360
* --static_image_mode<br>
静止画モード<br>
デフォルト：指定なし
* --model_complexity<br>
モデルの複雑度(0:Lite 1:Full 2:Heavy)<br>
※性能差は[Pose Estimation Quality](https://google.github.io/mediapipe/solutions/pose#pose-estimation-quality)を参照ください<br>
デフォルト：1
* --min_detection_confidence<br>
検出信頼値の閾値<br>
デフォルト：0.5
* --min_tracking_confidence<br>
トラッキング信頼値の閾値<br>
デフォルト：0.5
* --rev_color<br>
背景色とピクトグラムの色を反転する<br>
デフォルト：指定なし

# Author
高橋かずひと(https://twitter.com/KzhtTkhs)
 
# License 
Tokyo2020-Pictogram-using-MediaPipe is under [Apache-2.0 License](LICENSE).
