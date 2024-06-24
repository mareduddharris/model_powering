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
c.header("Models: Evaluating the model strength required with different prevelances of outcomes being predicted")
c.write("Use this work through to ascertain the effects different model strengths (ROC) will have on being able to predict an outcome")
c.write("Actions taken on a prediction can be positive or potentially negative in the incorrect patient group. This tool allows you to evaluate this impact based on outcome prevelance and model strength"