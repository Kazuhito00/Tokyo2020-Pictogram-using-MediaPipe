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

## Using Docker

Ubuntuの場合はホストマシンにMediaPipeをインストールせず、Docker + docker-composeを使うこともできます。

まず環境に合わせて`docker-compose.yml`を編集します。  
ビデオデバイスを指定する際`video0`を使う場合は以下のように編集します。

```diff
    # Edit here
    devices:
      # - "/dev/video0:/dev/video0"
      # - "/dev/video1:/dev/video0"
-     - "/dev/video2:/dev/video0"
+     - "/dev/video0:/dev/video0"
```

次にDockerイメージをビルドします。

```
docker-compose build
```

GUIアプリケーションの起動（X11 Forwarding）を許可します。

```
xhost +local:root
```

最後にDockerコンテナを起動します。

```
docker-compose up
```

# Author
高橋かずひと(https://twitter.com/KzhtTkhs)
 
# License 
Tokyo2020-Pictogram-using-MediaPipe is under [Apache-2.0 License](LICENSE).
