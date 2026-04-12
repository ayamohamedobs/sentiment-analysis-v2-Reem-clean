using 'main.bicep'

param baseName = 'sentiment'
param location = 'eastus2'
param gptModelName = 'gpt-5.4-mini'
param gptModelVersion = '2024-11-20'
param gptCapacity = 10
param tags = {
  project: 'sentiment-analysis'
  environment: 'dev'
}
