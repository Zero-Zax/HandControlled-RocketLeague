const express = require('express');
const path = require('path');
const app = express();

// Serve static files from the "public" directory.
app.use(express.static(path.join(__dirname, 'public')));

// Error feedback middleware for 404 errors.
app.use((req, res, next) => {
  res.status(404).send(`
    <h1>404 Not Found</h1>
    <p>The requested resource: <strong>${req.originalUrl}</strong> was not found on this server.</p>
    <hr>
    <p>
      Possible causes:
      <ul>
        <li>The file does not exist at the specified path.</li>
        <li>There is a typo or case mismatch in the file name or directory.</li>
        <li>The static file serving configuration in the server is incorrect.</li>
      </ul>
    </p>
    <p>Please verify that the file exists in the expected location and that the server is configured to serve it.</p>
  `);
});

app.listen(3000, () => {
  console.log('Server running on port 3000');
});
