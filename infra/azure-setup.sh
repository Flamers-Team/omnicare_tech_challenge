#!/usr/bin/env bash
# Configuração única do Azure para o OmniCare AI. Rode no Cloud Shell (Bash) do portal.
#
#   bash azure-setup.sh <nome-do-acr>
#
# O nome do ACR é global: só letras minúsculas e números, de 5 a 50 caracteres (ex.: omnicareacr2026).
# O script pede as chaves do Azure Speech e Language sem mostrá-las na tela nem gravá-las no histórico.
# Pode ser rodado de novo: o que já existe é reaproveitado.
set -euo pipefail

ACR_NAME="${1:?Informe o nome do ACR. Ex.: bash azure-setup.sh omnicareacr2026}"
RESOURCE_GROUP="RG_IA"
WEBAPP_NAME="omnicare"
LOCATION="westus3"                 # mesma região do app
GITHUB_APP_NAME="omnicare-github"  # app criado para o login do GitHub (OIDC)
SPEECH_REGION="eastus"
SPEECH_LANGUAGE="pt-BR"

SUB_ID=$(az account show --query id -o tsv)
RG_SCOPE="/subscriptions/${SUB_ID}/resourceGroups/${RESOURCE_GROUP}"

echo "==> Registry de imagens (${ACR_NAME})"
az acr show -n "$ACR_NAME" -g "$RESOURCE_GROUP" >/dev/null 2>&1 \
  || az acr create -n "$ACR_NAME" -g "$RESOURCE_GROUP" -l "$LOCATION" --sku Basic >/dev/null
ACR_ID=$(az acr show -n "$ACR_NAME" -g "$RESOURCE_GROUP" --query id -o tsv)

echo "==> Permissão do GitHub para enviar imagens (AcrPush)"
GH_APP_ID=$(az ad app list --display-name "$GITHUB_APP_NAME" --query "[0].appId" -o tsv)
if [ -z "$GH_APP_ID" ]; then
  echo "App '$GITHUB_APP_NAME' não encontrado. Crie o login do GitHub primeiro (passo de OIDC)." >&2
  exit 1
fi
az role assignment create --assignee "$GH_APP_ID" --role AcrPush --scope "$ACR_ID" >/dev/null 2>&1 || true

echo "==> Identidade do App Service para puxar a imagem (AcrPull)"
PRINCIPAL_ID=$(az webapp identity assign -g "$RESOURCE_GROUP" -n "$WEBAPP_NAME" --query principalId -o tsv)
az role assignment create --assignee-object-id "$PRINCIPAL_ID" --assignee-principal-type ServicePrincipal \
  --role AcrPull --scope "$ACR_ID" >/dev/null 2>&1 || true
az webapp config set -g "$RESOURCE_GROUP" -n "$WEBAPP_NAME" \
  --generic-configurations '{"acrUseManagedIdentityCreds": true}' --always-on true >/dev/null

echo "==> Configurações do app (porta, tempo de início, alertas persistentes)"
az webapp config appsettings set -g "$RESOURCE_GROUP" -n "$WEBAPP_NAME" --settings \
  WEBSITES_PORT=8000 \
  WEBSITES_CONTAINER_START_TIME_LIMIT=600 \
  WEBSITES_ENABLE_APP_SERVICE_STORAGE=true \
  OMNICARE_ALERTS_DB=/home/data/alertas.db \
  AZURE_SPEECH_REGION="$SPEECH_REGION" \
  AZURE_SPEECH_LANGUAGE="$SPEECH_LANGUAGE" >/dev/null

echo "==> Chaves do Azure (digite; nada aparece na tela)"
read -r -s -p "AZURE_SPEECH_KEY: " SPEECH_KEY; echo
read -r -s -p "AZURE_LANGUAGE_KEY: " LANG_KEY; echo
read -r -p "AZURE_LANGUAGE_ENDPOINT (https://...cognitiveservices.azure.com/): " LANG_ENDPOINT
az webapp config appsettings set -g "$RESOURCE_GROUP" -n "$WEBAPP_NAME" --settings \
  AZURE_SPEECH_KEY="$SPEECH_KEY" AZURE_LANGUAGE_KEY="$LANG_KEY" AZURE_LANGUAGE_ENDPOINT="$LANG_ENDPOINT" >/dev/null
unset SPEECH_KEY LANG_KEY

echo
echo "Pronto. No GitHub (Settings > Secrets and variables > Actions > Variables), crie:"
echo "  ACR_NAME = ${ACR_NAME}"
echo "E em Secrets (se ainda não criou): AZURE_CLIENT_ID, AZURE_TENANT_ID, AZURE_SUBSCRIPTION_ID."
echo "Valores: CLIENT_ID=${GH_APP_ID}  TENANT_ID=$(az account show --query tenantId -o tsv)  SUBSCRIPTION_ID=${SUB_ID}"
