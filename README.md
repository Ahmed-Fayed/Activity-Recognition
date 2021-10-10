# Activity-Recognition
Activity Recognition Python, Tensorflow &amp; Keras

Still working on it...

Classifying six activities:

1-	WALKING

2-	WALKING_UPSTAIRS

3-	WALKING_DOWNSTAIRS

4-	SITTING

5-	STANDING

6-	LAYING

The experiments have been carried out with a group of 30 volunteers within an age bracket of 19-48 years. Each person performed six activities (WALKING, WALKINGUPSTAIRS, WALKINGDOWNSTAIRS, SITTING, STANDING, LAYING) wearing a smartphone (Samsung Galaxy S II) on the waist. Using its embedded accelerometer and gyroscope, we captured 3-axial linear acceleration and 3-axial angular velocity at a constant rate of 50Hz. The experiments have been video-recorded to label the data manually. The obtained dataset has been randomly partitioned into two sets, where 70% of the volunteers was selected for generating the training data and 30% the test data.

The sensor signals (accelerometer and gyroscope) were pre-processed by applying noise filters and then sampled in fixed-width sliding windows of 2.56 sec and 50% overlap (128 readings/window). The sensor acceleration signal, which has gravitational and body motion components, was separated using a Butterworth low-pass filter into body acceleration and gravity. The gravitational force is assumed to have only low frequency components, therefore a filter with 0.3 Hz cutoff frequency was used. From each window, a vector of features was obtained by calculating variables from the time and frequency domain. See 'features_info.txt' for more details.


# For each record it is provided:
Triaxial acceleration from the accelerometer (total acceleration) and the estimated body acceleration.
Triaxial Angular velocity from the gyroscope.
A 561-feature vector with time and frequency domain variables.
Its activity label.
An identifier of the subject who carried out the experiment.


# Accuracy

![model_1 Accuracy](https://user-images.githubusercontent.com/31994329/136478387-6fe19607-f596-46c6-8163-e3ac6838189d.png)


# Loss

![model_1 Loss](https://user-images.githubusercontent.com/31994329/136478399-3c3492d6-c1b9-436c-9988-5ad9905b6905.png)


# Performance per Epoch (CSVLogger callback)

[model_1.csv](https://github.com/Ahmed-Fayed/Activity-Recognition/files/7307553/model_1.csv)



# Confusion Matrix

![Confusion Matrix](https://user-images.githubusercontent.com/31994329/136698718-37473e83-8d70-49d2-9ccd-bdb860cf75d7.png)


# Receiver Operating Characteristic (ROC)

![ROC](https://user-images.githubusercontent.com/31994329/136698763-f75fc075-2831-4799-81aa-e6f4ab04200b.png)




