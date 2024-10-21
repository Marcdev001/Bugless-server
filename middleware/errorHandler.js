function errorHandler(err, req, res, next) {
  console.error('Error:', err);

  if (err.name === 'ValidationError') {
    return res.status(400).json({ error: err.message });
  }

  if (err.name === 'CastError' && err.kind === 'ObjectId') {
    return res.status(400).json({ error: 'Invalid ID format' });
  }

  if (err.name === 'MulterError') {
    return res.status(400).json({ error: err.message });
  }

  res.status(500).json({ error: err.message || 'An unexpected error occurred' });
}

module.exports = { errorHandler };