// Run once locally: node scripts/get-refresh-token.js
// Prints a refresh_token — copy it into backend/.env as GOOGLE_REFRESH_TOKEN.
// Requires GOOGLE_CLIENT_ID / GOOGLE_CLIENT_SECRET / GOOGLE_REDIRECT_URI already
// set in backend/.env (redirect URI must be http://localhost:3000/oauth/callback
// for this local flow, and must match exactly what you registered in Google Cloud Console).

import 'dotenv/config';
import express from 'express';
import { google } from 'googleapis';
import open from 'open';

const oauth2Client = new google.auth.OAuth2(
  process.env.GOOGLE_CLIENT_ID,
  process.env.GOOGLE_CLIENT_SECRET,
  process.env.GOOGLE_REDIRECT_URI || 'http://localhost:3000/oauth/callback'
);

const SCOPES = [
  'https://www.googleapis.com/auth/drive.file',
  'https://www.googleapis.com/auth/drive.readonly',
];

const authUrl = oauth2Client.generateAuthUrl({
  access_type: 'offline', // required to get a refresh_token
  prompt: 'consent',      // forces a refresh_token even on repeat runs
  scope: SCOPES,
});

const app = express();
app.get('/oauth/callback', async (req, res) => {
  const { code } = req.query;
  try {
    const { tokens } = await oauth2Client.getToken(code);
    console.log('\n=== SUCCESS ===');
    console.log('Add this to backend/.env as GOOGLE_REFRESH_TOKEN:\n');
    console.log(tokens.refresh_token);
    console.log('\n===============\n');
    res.send('Done — check your terminal for the refresh token. You can close this tab.');
  } catch (e) {
    console.error(e);
    res.status(500).send('Something went wrong — check the terminal.');
  } finally {
    setTimeout(() => process.exit(0), 500);
  }
});

app.listen(3000, () => {
  console.log('\nOpening browser for Google authorization...');
  console.log('If it does not open automatically, visit this URL:\n');
  console.log(authUrl, '\n');
  open(authUrl);
});
