from prometheus_client import Counter, Histogram

SERVICE_INITIALIZATION_TIME = Histogram("service_initialization_time_seconds", "Time taken for service initialization in seconds")

HTTP_LATENCY = Histogram("http_requests_latency_seconds", "HTTP request latency in seconds", ["endpoint"])
REQUESTS_COUNT = Counter("http_requests_total", "Total number of HTTP requests", ["method", "endpoint", "status"])
