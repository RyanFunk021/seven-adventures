# Seven Adventures

Static single-page photo site, exported from a Claude artifact so it can be shared
with a plain public URL (Claude artifacts require the viewer to be signed into the
right account to see them).

## Deploy on Render

1. Push this folder to a GitHub repo (or a subfolder of one).
2. In Render: New -> Static Site -> point at the repo.
   - Build command: (leave empty)
   - Publish directory: `.`
3. Render will give you a public `https://<name>.onrender.com` URL — send that to friends.

`render.yaml` is included for Render's "Blueprint" deploy option, which reads this
file automatically if you connect the repo via New -> Blueprint instead.

## Updating

If you edit the trip in Claude and republish the artifact, re-export it and replace
`index.html` here, then push — Render redeploys automatically on push.
