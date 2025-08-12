#!/bin/bash

gcloud builds submit --tag gcr.io/cel-streamlit/mechafil-jax-web --no-cache

gcloud run deploy mechafil-jax-web \
  --image gcr.io/cel-streamlit/mechafil-jax-web \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --min-instances=0 \
  --port=8501
