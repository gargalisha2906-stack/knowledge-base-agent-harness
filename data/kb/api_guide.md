# API Guide

The REST API base URL is https://api.northwind.example/v1. Authenticate with a
bearer token created under Settings > Developers. Tokens are shown once at
creation; store them like passwords. Rate limits: 60 requests per minute on
Starter, 1,000 on Growth and Scale. Exceeding the limit returns HTTP 429 with a
Retry-After header; wait that many seconds before retrying.

All list endpoints paginate with cursor parameters: pass the `next_cursor`
value from each response to get the next page. Responses are JSON. Errors use
RFC 7807 problem+json with a stable `type` field safe to branch on.

Webhooks: subscribe to ticket.created, ticket.closed, and invoice.paid events.
Deliveries time out after 10 seconds; failed deliveries retry 5 times with
exponential backoff over about 1 hour. Respond with any 2xx within 10 seconds
to acknowledge. If your endpoint is down, pause the subscription instead of
letting retries stack up. Verify webhook signatures with the signing secret
shown in Settings > Developers; never trust an unsigned payload.
