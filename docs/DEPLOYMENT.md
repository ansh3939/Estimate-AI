# Deployment Guide

## Prerequisites

### Environment Variables
Set the following environment variables in your deployment environment:

```bash
DATABASE_URL=postgresql://user:password@host:port/database
OPENAI_API_KEY=your_openai_api_key
PGHOST=your_pg_host
PGPORT=your_pg_port
PGUSER=your_pg_user
PGPASSWORD=your_pg_password
PGDATABASE=your_pg_database
```

### System Requirements
- Python 3.11+
- PostgreSQL 12+
- Minimum 2GB RAM
- 1GB storage space

## Replit Deployment

### Automatic Deployment
The application is pre-configured for Replit Deployments:

1. Click the "Deploy" button in your Replit interface
2. Select "Autoscale" deployment option
3. Configure custom domain (optional)
4. Monitor deployment status

### Configuration
The `.replit` file contains optimized deployment settings:

```toml
[deployment]
run = ["streamlit", "run", "src/main.py", "--server.port", "5000"]
```

### Health Checks
Replit automatically monitors:
- Application response time (<500ms)
- Memory usage
- Database connectivity
- SSL certificate validity

## Manual Deployment

### Local Development
```bash
# Clone repository
git clone <repository-url>
cd ai-real-estate-platform

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export DATABASE_URL="your_database_url"
export OPENAI_API_KEY="your_api_key"

# Run application
streamlit run src/main.py --server.port 5000
```

### Docker Deployment
```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 5000

CMD ["streamlit", "run", "src/main.py", "--server.port", "5000"]
```

## Production Optimization

### Performance Tuning
- Enable Streamlit caching for database queries
- Configure PostgreSQL connection pooling
- Implement CDN for static assets
- Enable gzip compression

### Security
- SSL/TLS encryption (automatic with Replit)
- Environment variable protection
- Database connection security
- API rate limiting

### Monitoring
- Application performance metrics
- Database query performance
- User session analytics
- Error tracking and logging

## Scaling Considerations

### Database Scaling
- Connection pooling configuration
- Read replica setup for analytics
- Database indexing optimization

### Application Scaling
- Horizontal pod autoscaling
- Load balancer configuration
- Session state management
- Cache layer implementation

## Troubleshooting

### Common Issues
1. **Import Errors**: Ensure all `__init__.py` files are present
2. **Database Connection**: Verify environment variables
3. **Memory Issues**: Monitor ML model loading
4. **Slow Performance**: Check database query optimization

### Debug Mode
Enable debug logging by setting:
```bash
export STREAMLIT_LOGGER_LEVEL=debug
```