# Mood Theremin

**Description:**

The theremin is an electronic musical instrument invented in the 1920s. It's played without physical contact—pitch and volume are controlled by moving hands near two antennas. The proximity of the right hand to the pitch antenna determines the pitch, while the left hand's distance from the volume antenna controls the volume. Known for its eerie sound, the theremin has been used in various music genres and movie soundtracks.

Inspired by the theremin, we created Mood Theremin in [Cal Hack 10.0](https://www.calhacks.io/). This is a unique project that combines facial expression analysis, emotion sentiment analysis, machine learning, and audio synthesis to generate music based on users' facial expressions. By utilizing the HUME streaming API for emotion sentiment analysis and a Scikit-Learn K-Nearest Neighbors (KNN) regressor model, we translate users' facial expressions into chords and corresponding volume levels. The generated chords are then played using the Web Audio API, creating a musical experience that mirrors users' emotions.

**How It Works:**

1. **Facial Expression Analysis:** Users' facial expressions are captured through video input.
2. **Emotion Sentiment Analysis:** The HUME streaming API is used to analyze the emotion sentiments expressed in the users' facial expressions.
3. **Machine Learning:** The emotion sentiment data is fed into a Scikit-Learn KNN regressor model to predict chords and volume levels based on emotions detected.
4. **Audio Synthesis:** The predicted chords are played using the Web Audio API, creating music in real-time.

**Requirements:**

- Python (3.6+)
- Scikit-Learn Package
- HUME streaming API access
- Web Audio API compatible web browser


**Usage:**

1. Obtain HUME streaming API credentials and configure the API access.
2. Run the facial expression analysis component to capture users' facial expressions via video input.
3. Process the facial expression data using the HUME streaming API to obtain emotion sentiment analysis results.
4. Feed the emotion sentiment data into the trained Scikit-Learn KNN regressor model to predict chords and volume levels.
5. Utilize the Web Audio API to play the generated chords with corresponding volume levels, creating emotion-driven music.

**Contribution:**

- Shujing Hu: Developed the machine learning model and fine-tuned the mapping algorithm.
- Tri Pham: Designed the front-end interactive 3-D experience using Three.js



**Acknowledgements:**

- [HUME streaming API](https://dev.hume.ai/docs/streaming-api) for emotion sentiment analysis.
- Scikit-Learn and Web Audio API communities for their valuable tools and resources.

Feel free to customize, expand, and explore the Emotion-Driven Music Generator project to create interactive and emotionally engaging musical experiences!


# TODO
- Add your contributions!!
- Flask and JSON (add more description?)
