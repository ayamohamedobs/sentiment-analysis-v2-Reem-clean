// Provisions an Azure AI Foundry project for sentiment analysis
// Resources: AI Services (Foundry), Project, GPT-4o deployment, AI Language
// All service-to-service auth uses managed identity (no API keys)
// Data source: local files (default) or Microsoft Fabric Data Agent (set enableFabric = true)

targetScope = 'resourceGroup'

// ─── Parameters ─────────────────────────────────────────────────────────────

@description('Base name used to derive resource names')
@minLength(3)
param baseName string

@description('Azure region for all resources')
param location string = resourceGroup().location

@description('GPT model to deploy as the agent backbone')
param gptModelName string = 'gpt-4o'

@description('GPT model version')
param gptModelVersion string = '2024-05-13'

@description('Capacity (TPM in thousands) for the GPT deployment')
param gptCapacity int = 50

@description('Tags applied to every resource')
param tags object = {}

@description('Enable Microsoft Fabric Data Agent connection as data source (true/false)')
param enableFabric string = 'false'

var fabricEnabled = enableFabric == 'true'

@description('Fabric connection name in the Foundry project')
param fabricConnectionName string = 'fabric-rsa-survey'

@description('Microsoft Fabric workspace ID (required when enableFabric = true)')
param fabricWorkspaceId string = ''

@description('Fabric Data Agent artifact ID (required when enableFabric = true)')
param fabricArtifactId string = ''

@description('Azure region for the Fabric capacity (defaults to main location)')
param fabricLocation string = location

@description('Fabric capacity SKU (e.g. F2, F4, F8)')
param fabricSkuName string = 'F2'

@description('Fabric capacity admin email')
param fabricAdminEmail string = ''

// ─── Derived names ──────────────────────────────────────────────────────────

var uniqueSuffix = uniqueString(resourceGroup().id, baseName)
var aiServicesName = '${baseName}-ais-${uniqueSuffix}'
var languageName = '${baseName}-lang-${uniqueSuffix}'
var logAnalyticsName = '${baseName}-logs-${uniqueSuffix}'
var appInsightsName = '${baseName}-appins-${uniqueSuffix}'
var fabricCapacityName = '${replace(toLower(baseName), '-', '')}fabric${uniqueSuffix}'
var projectName = 'sentiment-analysis'

// ─── Well-known role definition IDs ─────────────────────────────────────────

// ─── Log Analytics Workspace (for Application Insights) ────────────────────

resource logAnalytics 'Microsoft.OperationalInsights/workspaces@2023-09-01' = {
  name: logAnalyticsName
  location: location
  tags: tags
  properties: {
    sku: {
      name: 'PerGB2018'
    }
    retentionInDays: 30
    features: {
      enableLogAccessUsingOnlyResourcePermissions: true
    }
  }
}

// ─── Application Insights (for agent monitoring & telemetry) ───────────────

resource appInsights 'Microsoft.Insights/components@2020-02-02' = {
  name: appInsightsName
  location: location
  tags: tags
  kind: 'web'
  properties: {
    Application_Type: 'web'
    WorkspaceResourceId: logAnalytics.id
    IngestionMode: 'LogAnalytics'
    publicNetworkAccessForIngestion: 'Enabled'
    publicNetworkAccessForQuery: 'Enabled'
  }
}

// ─── Diagnostic Settings (sends AI Services telemetry to App Insights) ─────

resource aiServicesDiagnostics 'Microsoft.Insights/diagnosticSettings@2021-05-01-preview' = {
  name: 'send-to-appinsights'
  scope: aiServices
  properties: {
    workspaceId: logAnalytics.id
    logs: [
      {
        categoryGroup: 'allLogs'
        enabled: true
      }
    ]
    metrics: [
      {
        category: 'AllMetrics'
        enabled: true
      }
    ]
  }
}

// ─── Azure AI Services account (hosts the Foundry project) ─────────────────

resource aiServices 'Microsoft.CognitiveServices/accounts@2025-06-01' = {
  name: aiServicesName
  location: location
  tags: tags
  kind: 'AIServices'
  identity: {
    type: 'SystemAssigned'
  }
  sku: {
    name: 'S0'
  }
  properties: {
    customSubDomainName: aiServicesName
    publicNetworkAccess: 'Enabled'
    disableLocalAuth: true // enforce managed identity, no API keys
    allowProjectManagement: true
    networkAcls: {
      defaultAction: 'Allow'
    }
  }
}

// ─── Foundry Project ────────────────────────────────────────────────────────

resource foundryProject 'Microsoft.CognitiveServices/accounts/projects@2025-06-01' = {
  parent: aiServices
  name: projectName
  location: location
  tags: tags
  identity: {
    type: 'SystemAssigned'
  }
  properties: {
    displayName: 'Sentiment Analysis Agent'
    description: 'AI Foundry project that hosts an agent for analysing survey sentiment using Azure AI Language.'
  }
}

// ─── GPT-4o Model Deployment (agent backbone) ──────────────────────────────

resource gptDeployment 'Microsoft.CognitiveServices/accounts/deployments@2025-06-01' = {
  parent: aiServices
  name: 'gpt-4o'
  sku: {
    name: 'GlobalStandard'
    capacity: gptCapacity
  }
  properties: {
    model: {
      format: 'OpenAI'
      name: gptModelName
      version: gptModelVersion
    }
    versionUpgradeOption: 'OnceNewDefaultVersionAvailable'
  }
}

// ─── Azure AI Language (Text Analytics / Sentiment) ────────────────────────

resource languageService 'Microsoft.CognitiveServices/accounts@2025-06-01' = {
  name: languageName
  location: location
  tags: tags
  kind: 'TextAnalytics'
  identity: {
    type: 'SystemAssigned'
  }
  sku: {
    name: 'S'
  }
  properties: {
    customSubDomainName: languageName
    publicNetworkAccess: 'Enabled'
    disableLocalAuth: true // enforce managed identity, no API keys
    networkAcls: {
      defaultAction: 'Allow'
    }
  }
}

// ─── Project connection to Language service (AAD / managed identity) ───────

resource languageConnection 'Microsoft.CognitiveServices/accounts/projects/connections@2025-06-01' = {
  parent: foundryProject
  name: 'language-sentiment'
  properties: {
    authType: 'AAD'
    category: 'CognitiveService'
    target: languageService.properties.endpoint
    useWorkspaceManagedIdentity: true
    metadata: {
      Kind: 'AIServices'
      ApiType: 'azure'
      ResourceId: languageService.id
    }
  }
}

// ─── Project connection to Application Insights ────────────────────────────

resource appInsightsConnection 'Microsoft.CognitiveServices/accounts/projects/connections@2025-06-01' = {
  parent: foundryProject
  name: 'appInsights-connection'
  properties: {
    authType: 'ApiKey'
    category: 'AppInsights'
    target: appInsights.id
    metadata: {
      ApiType: 'Azure'
      ResourceId: appInsights.id
    }
    credentials: {
      key: appInsights.properties.InstrumentationKey
    }
  }
}

// ─── Microsoft Fabric Capacity (opt-in) ────────────────────────────────────

resource fabricCapacity 'Microsoft.Fabric/capacities@2023-11-01' = if (fabricEnabled) {
  name: fabricCapacityName
  location: fabricLocation
  tags: tags
  sku: {
    name: fabricSkuName
    tier: 'Fabric'
  }
  properties: {
    administration: {
      members: [
        fabricAdminEmail
      ]
    }
  }
}

// ─── Fabric Data Agent connection (opt-in) ─────────────────────────────────
// Flip enableFabric = true and provide fabricWorkspaceId to connect
// Source Data → Fabric Semantic Model → Fabric Data Agent → this Foundry Agent

resource fabricConnection 'Microsoft.CognitiveServices/accounts/projects/connections@2025-06-01' = if (fabricEnabled) {
  parent: foundryProject
  name: fabricConnectionName
  properties: {
    authType: 'CustomKeys'
    category: 'CustomKeys'
    target: '_'
    metadata: {
      type: 'fabric_dataagent'
      'workspace-id': fabricWorkspaceId
      'artifact-id': fabricArtifactId
    }
    credentials: {
      keys: {
        placeholder: 'none'
      }
    }
  }
}


// ─── Outputs ────────────────────────────────────────────────────────────────

@description('AI Services endpoint')
output aiServicesEndpoint string = aiServices.properties.endpoint

@description('Foundry project endpoint for agent API calls')
output foundryProjectEndpoint string = '${aiServices.properties.endpoint}api/projects/${projectName}'

@description('AI Language endpoint for sentiment analysis')
output languageEndpoint string = languageService.properties.endpoint

@description('Fabric connection name (empty when Fabric is disabled)')
output fabricConnectionName string = fabricEnabled ? fabricConnectionName : ''

@description('Resource group name')
output resourceGroupName string = resourceGroup().name

@description('GPT deployment name')
output gptDeploymentName string = gptDeployment.name

@description('Application Insights connection string')
output appInsightsConnectionString string = appInsights.properties.ConnectionString

@description('Application Insights instrumentation key')
output appInsightsInstrumentationKey string = appInsights.properties.InstrumentationKey
