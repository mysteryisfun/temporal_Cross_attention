# ChaLearn First Impressions V2 Dataset Structure

The **ChaLearn First Impressions V2** dataset is designed for the task of apparent personality analysis from short video clips. It combines high-quality annotated audiovisual data with semantic transcriptions and demographic labels, enabling multimodal learning.

---

## 1. Dataset Overview

* **Total Videos:** 10,000
* **Average Duration:** \~15 seconds
* **Source:** YouTube HD vlogs with individuals speaking directly to the camera
* **Diversity:** Varying gender, age, nationality, and ethnicity
* **Language:** English
* **Split Ratio:**

  * Training: 60% (6000 videos)
  * Validation: 20% (2000 videos)
  * Test: 20% (2000 videos)

---

## 2. Data Modalities

### a. **Video Data**

* Each clip contains a speaker talking to the camera.
* Used for visual (static + dynamic) and audio cues.
* Provided as zipped sets (e.g., `train-1.zip` to `train-6.zip`)

### b. **Transcriptions**

* Provided per video.
* Stored as a single pickled Python dictionary:

  ```python
  transcription['video_id']  # Returns the transcription string
  ```
* Average: 43 words/video (18 non-stopwords)

### c. **Annotations (Labels)**

* Six numerical labels per video, range: \[0.0, 1.0]:

  * Big Five Personality Traits:

    * Openness
    * Conscientiousness
    * Extraversion
    * Agreeableness
    * Neuroticism
  * Interview Suitability Score ("interview")
* Format:

  ```python
  annotation['trait']['video_id']  # e.g., annotation['openness']['abc123']
  ```

---

## 3. Demographic Annotations (Supplementary)

* **Gender:** Male = 1, Female = 2
* **Ethnicity:**

  * 1 = Asian
  * 2 = Caucasian
  * 3 = African-American
* **Age Ranges (Group ID):**

  * 1: \[0–6]
  * 2: \[7–13]
  * 3: \[14–18]
  * 4: \[19–24]
  * 5: \[25–32]
  * 6: \[33–45]
  * 7: \[46–60]
  * 8: 61+

Files:

* `eth_gender_annotations_dev.csv`
* `eth_gender_annotations_test.csv`

---

## 4. Additional Labels

* **Pairwise Annotations:** Raw relative comparison labels between clips.
* **Predicted Attributes:** Attractiveness and perceived age (soft labels).

---

## 5. Ground Truth and Prediction Files

* **Annotations** and **transcriptions** are stored as pickle files.
* Sample prediction files are provided for quantitative (test) and qualitative phases.

### Example File Access

```python
import pickle

with open('train-annotation.pkl', 'rb') as f:
    annotation = pickle.load(f)
    openness_score = annotation['openness']['video_id']

with open('train-transcription.pkl', 'rb') as f:
    transcription = pickle.load(f)
    transcript_text = transcription['video_id']
```

---

## 6. Encryption Keys

Some files require decryption:

* Validation/test groundtruth: `zeAzLQN7DnSIexQukc9W`
* Test 80 zip files: `.chalearnLAPFirstImpressionsSECONDRoundICPRWorkshop2016.`

---

## 7. Citation References

* Biel, Aran, Gatica-Perez (2011, 2013)
* Escalante et al., IEEE Trans. on Affective Computing (TAC), 2020

---

## 8. Download Links

Refer to the dataset website or use direct links to access:

* Videos (`train-*.zip`, `val-*.zip`, `test-*.zip`)
* Annotations (`*-annotation.zip`)
* Transcriptions (`*-transcription.zip`)
* Demographics (`eth_gender_annotations_*.csv`, `fi_age_labels.zip`)

---

This structure is foundational for building multimodal personality recognition models by fusing facial features, motion dynamics, speech content, and demographic indicators.
