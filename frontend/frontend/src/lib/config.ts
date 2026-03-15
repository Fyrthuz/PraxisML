// Configuration for API URLs
// Uses environment variables for deployment flexibility

const getApiBaseUrl = (): string => {
  // Priority: Environment variable > Default
  const baseUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
  // Remove trailing slash if present
  return baseUrl.replace(/\/$/, "");
};

const getWsBaseUrl = (): string => {
  // For WebSocket, we need to convert http to ws
  const httpUrl = getApiBaseUrl();
  return httpUrl.replace(/^http/, "ws");
};

export const config = {
  API_BASE_URL: getApiBaseUrl(),
  WS_BASE_URL: getWsBaseUrl(),
  API_VERSION: "/api/v1",
  getFullApiUrl: (endpoint: string): string => {
    return `${getApiBaseUrl()}/api/v1${endpoint}`;
  },
  getFullWsUrl: (endpoint: string): string => {
    return `${getWsBaseUrl()}/api/v1${endpoint}`;
  },
};