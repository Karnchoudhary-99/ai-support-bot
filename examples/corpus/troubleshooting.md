---
title: Troubleshooting Deployments
tags: troubleshooting, deployments
---

# Troubleshooting Deployments

## Build failures

If your build fails, first check the build log in the dashboard under
**Deployments → Logs**. The most common causes are a missing lockfile
(`package-lock.json` or `requirements.txt`) or a Node version mismatch.
Pin your runtime in `acme.json` with the `runtime` field.

## Deployment stuck in "Queued"

Deployments queue when your workspace has exhausted its concurrent build
slots. Starter workspaces get 1 slot, Pro gets 5, Enterprise is unlimited.
Cancel stuck deployments from **Deployments → Cancel** and retry.

## 502 errors after deploy

A 502 right after deploy usually means your app crashed on boot or is not
listening on the `$PORT` environment variable. Acme routes traffic to the
port in `$PORT` — hardcoding port 3000 or 8080 is the most common mistake.
Check **Deployments → Logs** for the crash message.

## Rollbacks

Every successful deployment is kept for 30 days. Roll back instantly from
**Deployments → History → Roll back**. Rollbacks do not trigger a rebuild.
