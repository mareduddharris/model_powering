# SINGLE-CLASS PREDICTION + INTERVENTION
#
# This page is supposed to be an end-to-end look at the
# effectiveness of a model designed to predict one binary
# class, and combine it with the chance of success
# of some action taken as a result of the prediction.
#
# The ROC AUC only shows true/false positive rates, which
# does not included the prevalence of the item being
# predicted.
#
# By including the prevalence, the model performance can
# be converted to positive/negative predictive values,
# which is the chance the model prediction is right when
# it is used in practice.
#
# If some action is taken when the model predicts
# a _positive_ class, and the probability of success is provided,
# it is possible to calculate how often the model will
# result in a successful outcome. This is a practical
# metric of interest.
#
# If, in addition, the probability of harm is specified for
# when the action is (incorrectly) taken on a _negative_
# class, it is possible to work out the proportion of
# patients who would be harmed by the use of the model.
#
# This is intended as a simple introduction to the calculations
# involved in more complicated cases (e.g. when two classes
# are being predicted, and the action trades off between two
# adverse outcomes)
#
# What we need for this page to make sense is a single class
# which has these criteria:
# 
# * Something about a patient that can be predicted by a model
# * Something about a patient that has a background prevalence
# * Something where some action can be taken based on knowing the class
#   for the patient
# * Something where doing the action on a patient with the class has
#   a probability of success, and doing it on a patient without
#   the class has a probability of harm.

# eg. images of skin cancer, images in radiology.

import utils
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd

st.title("Your title here...")

# STEP 0 -- GETTING UP AND RUNNING
#
# Copy this file and put it in your pages/ folder. Then open a
# terminal (Terminal -> New Terminal) and run:
#
#     streamlit run Overview.py
#
# Open your browser to localhost:8501. Make any change in the file,
# and save it. Then navigate back to the page and click 
# `Always Rerun`.

# STEP ONE -- BASELINE CLASS PREVALENCE
#
# The first step is to get the background prevalence of the class
# being predicted
#
# We are interested in just one class being predicted, so there is 
# just one number here, p_positive, which is the baseline probability
# of a positive case for any patient.
#
# TASK
#
# Currently, this is just a constant. Change this to something
# in streamlit (e.g. c.number_input) that gets the number from the
# user.
c = st.container(border=True)
c.header("Baseline Prevalence", divider=True)
c.write("TODO: get the baseline prevalence from the user")
p_positive = st.number_input("Background Prevalence (%)", min_value=0, max_value=100)/100
st.write("The assigned background prevelance (%) is ", p_positive*100)


# STEP TWO -- MODEL PERFORMANCE
#
# p_b above is the background prevalence -- the next step is for the user
# to specify the model performance.
#
# This is the first important bit of stats: for the purposes of this page,
# the model performance is specified by giving a true positive rate (TPR) and
# a true negative rate (TNR).
#
# Quick Reminder: These true/false positive/negative rates are probabilities
# that the model will be right, _given that the patient definitely is/isn't in
# the class_. This is last bit is what makes the number inapplicable to real
# life, because a given patient is not definitely in one or other of these
# categories (though depending on the use case they actually really might be - 
# eg. there is thrombus. If a risk predictor then obviously can't be certain around 
#  a risk score)
#
# TIP: Once you know the TPR and the TNR, you also know the FPR and FNR
# (the formulas are FPR = 1 - TNR, FNR = 1 - TPR). We do everything in terms
# of true-rates because it feels more intuitive to work with. Note that
# the TPR and TNR are independent -- you need to specify both. One says
# how good the model is for positive patients, and the other says how good
# it is for negative patients.
#
# TASK
#
# Get the user to input the TPR and TNR using something like st.number_input.
# Keep it simple to begin with -- you can do it exactly the same as the
# one above for the prevalence. 
#
# BONUS/TRICKY: Later (after you've done everything else), come back and think how
# to use c.toggle to let the user choose if they want to put in fpr and
# fnr instead. Have a look at the model_accuracy() function in inputs.py for
# inspiration. Remember to convert these into tpr and tnr for the script, otherwise
# everything below will have to change too.
#
c = st.container(border=True)
c.header("Model Performance", divider=True)

tpr = st.number_input("True Positive Rate(%)", min_value=0, max_value=100)/100
st.write(f"The assigned True Positive Rate is {tpr*100}(%)")

tnr = st.number_input("True Negative Rate(%)", min_value=0, max_value=100)/100
st.write(f"The assigned True Negative Rate is {tnr*100}(%)")

# Get the other rates for convenience
fpr = 1 - tnr
fnr = 1 - tpr

# STEP THREE -- PLOTTING MODEL ROC
#
# You can make a really simple link between the tpr and tnr above
# and the model's ROC curve, by drawing a ROC curve out of straight
# lines that passes through the right point.
#
# The x-axis of a ROC curve has fpr, and the y-axis as tpr. We are
# going to calculate the coordinates of the point the ROC curve must
# pass through, which is (fpr, tpr).
#
# Since we only have tpr and tnr, we need to get the fpr using 
# fpr = 1 - tnr. 
# As a result, we need to plot a straight line
# graph going from (0,0) to (1 - tnr, tpr), and thn to (1,1)
#
# TASK
#
# Fix the code below to plot the required graph. The framework
# is in place, but the important middle coordinate is missing.
# (Hint: you need to add something to the x and y lists)
#
# BONUS: Add axis labels, and look into scaling the x and y
# axes so they are probabilities (0 to 100%). Have a look at
# roc.py for inspiration.
#

fig, ax = plt.subplots()

# Calculate the auc, which is just the area of the
# triangle -- you can use utils.auc (look at utils.py)
auc = utils.simple_auc(tpr, tnr)

# List of x and y coordinates to plot. 
# The relevant point is missing..
x = [0.0,fpr, 1.0]
y = [0.0, tpr, 1.0]
ax.plot(x, y, label=f"The AUC is {auc:.2f}")
ax.legend()

c.pyplot(fig)

# STEP 4 -- EFFECT OF ACTION
#
# When the model predicts a positive class, an 
# action is performed. In this simple version of
# the calculation, the action has a probability
# of success when applied to the correct patients
# (i.e. when applied to patients who really are
# in the positive class), but has a chance of harm
# when applied to the wrong patient.
#
# (Note: you can set the chance of harm to 0 -- that
# might apply to actions which have only a positive
# effect when applied to the correct patients, but
# just don't do anything when applied incorrectly.
# An example might be triggering doctors to look 
# more closely at medical images? Can have a think if
# there's something better/different to do here.)
#
# TASK
# 
# We need two numbers -- the probability of successful
# action when applied correctly (p_success), and the
# probability of harm when applied incorrectly (p_harm).
# Get these probabilities from the user
# 
c = st.container(border=True)
c.header("Effect of Action", divider=True)
c.write("There are often actions taken based on a positive test result. In the correct (True Positive) cohort this can have a successful outome. However, in an incorrect (False Positive) cohort this could be harmful - eg. Reducing blood thinners.")

p_success = st.number_input("In a true positive test result, what is the chance of an action being successful", min_value=0, max_value=100)/100
st.write(f"The assigned Rate of successful action based on true positive test result is {p_success*100}(%)")

p_harm = st.number_input("In a false positive test result, what is the chance of the same action causing harm. This could be zero depending on use case.", min_value=0, max_value=100)/100
st.write(f"The assigned rate of harm through action taken based on a false positive test result is {p_harm*100}(%)")

# STEP 5 -- MATHS TIME
#
# The purpose of the above steps was to collect the following
# bits of information from the user:
#
# * Background prevalence of positive outcome: p_positive
#
# * Model accuracy when the patient is definitely positive: tpr
# * Model accuracy when the patient is definitely negative: tnr
#
# * Chance of action-success when applied to positive patient: p_success
# * Chance of action-harm when applied to negative patient: p_harm
#
# We will combine these to derive the probability of success
# and the probability of harm in an end-to-end use of the model
# for triggering an action
#

# The main idea of the calculation is to consider how successful
# and non-successful outcomes can happen. For example, one way
# that a successful outcome can happen is:
#
# 1) Patient is really positive
# 2) Patient is predicted positive, (therefore) action is taken
# 3) Action was successful
# 
# The probability of this sequence of events is obtained by
# multiplying the probabilities of each step:


A = p_positive * tpr * p_success

# Breaking this down further -- p_positive is the probability 
# of the patient being positive; tpr is the probability of
# the model correctly predicting them as positive (we can
# use tpr because we are assuming the patient is positive here),
# and p_success is the probability of action success.
#
# In this model, this is the only way action-success can happen,
# and so A is the probability of end-to-end success.

# TO THINK ABOUT: Should "success" also include the case where
# the patient is negative, the model predicts them as negative,
# and no action is taken? Currently, we are lumping that into
# neutral (see below). If you think so, calculate the probability
# of this in the same way as above, and add it to A. 
#
# I think exactly what to do would be clarified by a few examples,
# if we can think of some.

# TASK -- WORK OUT PROBABILITY OF HARM

# 
#
# The probability of harm similarly has exactly one way it
# can occur -- fill it in below:
#
# 1) Patient's real class is negative (1 - p_positive)
# 2) Model has to predict positive when patient is negative (fpr)
# 3) Model has to cause harm (p_harm)
#
# As a result, write down the calculation for the probability
# that harm occurs:
# B = 0.0 * 0.0 * 0.0 # There will be three probabilities multiplied
B = (1-p_positive) * fpr * p_harm

# There are other things that happen -- in our simple model, we
# have not considered what happens if cases such as when the model
# makes a negative prediction, so action is not taken -- we are not
# classifying this as harm currently, we are just saying it is 
# neutral.
#
# The probability of a neutral result is just whatever is left s
# after subtracting the probabilities of success and harm.
#
C = 1.0 - A - B

# STEP 6 -- PRESENT ALL THE RESULTS
#
# You can make a final box which shows what the breakdown of the
# results in terms of success, harm and neutral (maybe a pie chart?)
#
c = st.container(border=True)
c.header("Results", divider=True)
c.write("You can present all the results here")

# Define the three values
data = {
    'Category': ['Probability of Success', 'Probability of Harm','Neither'],
    'Value': [A, B, C]
}

# Create a DataFrame from the data
df = pd.DataFrame(data).set_index("Category")
print(df)
# Display the data in a table
st.write("Data in Table:")
st.write((100*df).style.format("{:.2f}%"))

# Create a pie chart
fig, ax = plt.subplots()
ax.pie(df['Value'], labels=df.index, autopct='%1.1f%%', startangle=90)
ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
st.write("Pie Chart:")
st.pyplot(fig)

