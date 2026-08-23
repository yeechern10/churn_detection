from fastapi import Request
from prometheus_client import Counter, Histogram

from app.prom_metrics import HTTP_LATENCY, REQUESTS_COUNT

from starlette.middleware.base import BaseHTTPMiddleware

class PrometheusMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        status = 500  # Default status code
        
        # avoid tracking metrics for the /metrics endpoint to prevent infinite recursion
        if request.url.path == "/metrics":
            return await call_next(request)

        
        try:
            with HTTP_LATENCY.labels(endpoint=request.url.path).time():
                response = await call_next(request)

                status = response.status_code

            return response

        except Exception:
            status = 500
            raise

        finally:

            REQUESTS_COUNT.labels(
                method=request.method,
                endpoint=request.url.path,
                status=str(status)
            ).inc()
