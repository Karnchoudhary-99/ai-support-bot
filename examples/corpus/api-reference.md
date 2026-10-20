---
title: API Reference
tags: api, developers
---

# API Reference

Base URL: `https://api.acme.example.com/v1`. All requests need a bearer
token from **Settings → API Keys**.

## Authentication

Pass your key as `Authorization: Bearer <key>`. Keys start with `acme_`
and can be scoped read-only. Rotate keys every 90 days.

## Rate limits

The API allows 100 requests per minute per key. Exceeding the limit returns
HTTP 429 with a `Retry-After` header in seconds. Enterprise workspaces can
request higher limits via support.

## Webhooks

Configure webhooks in **Settings → Webhooks**. We POST JSON to your URL on
`deploy.succeeded`, `deploy.failed`, and `billing.payment_failed` events.
Webhook deliveries are retried with exponential backoff for 24 hours.
