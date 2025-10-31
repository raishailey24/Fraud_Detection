# AWS EC2 Deployment Guide for FraudSight (2.97GB Dataset)

## 🚀 Quick Deployment Steps

### 1. Launch EC2 Instance
```bash
# Recommended Instance: t3.large (2 vCPU, 8GB RAM)
# OS: Ubuntu 22.04 LTS
# Storage: 20GB SSD (for OS + dataset)
# Security Group: Allow port 8501
```

### 2. Connect and Setup
```bash
# Connect to your instance
ssh -i your-key.pem ubuntu@your-ec2-ip

# Update system
sudo apt update && sudo apt upgrade -y

# Install Docker
sudo apt install docker.io docker-compose -y
sudo usermod -aG docker ubuntu
sudo systemctl start docker
sudo systemctl enable docker

# Logout and login again for docker group to take effect
exit
```

### 3. Deploy FraudSight
```bash
# Clone your repository or upload files
git clone <your-repo> fraudsight
cd fraudsight

# Copy your dataset to data folder
# Upload complete_user_transactions.csv to ./data/

# Set environment variables
cp .env.example .env
# Edit .env with your API keys

# Deploy with Docker
docker-compose up -d

# Check logs
docker-compose logs -f
```

### 4. Access Dashboard
```
http://your-ec2-public-ip:8501
```

## 💰 Cost Estimation

| Instance Type | RAM | vCPU | Monthly Cost |
|---------------|-----|------|--------------|
| t3.large      | 8GB | 2    | ~$60-80      |
| t3.xlarge     | 16GB| 4    | ~$120-150    |
| m5.large      | 8GB | 2    | ~$70-90      |

## ⚡ Performance Optimizations

### Memory Settings
```bash
# Add to docker-compose.yml
environment:
  - PYTHONUNBUFFERED=1
  - STREAMLIT_SERVER_MAX_UPLOAD_SIZE=3000
  - STREAMLIT_SERVER_MAX_MESSAGE_SIZE=3000
```

### Instance Optimization
```bash
# Increase swap space for large dataset loading
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

## 🔧 Troubleshooting

### If Memory Issues:
1. Use t3.xlarge (16GB RAM) instead
2. Reduce data limit in dashboard
3. Monitor with `docker stats`

### If Loading Slow:
1. Check disk I/O with `iostat`
2. Consider using SSD storage
3. Monitor with `htop`

## 🛡️ Security

### Basic Security Setup:
```bash
# Update security group to restrict access
# Only allow your IP on port 8501
# Use HTTPS with nginx proxy (optional)

# Setup basic firewall
sudo ufw allow ssh
sudo ufw allow 8501
sudo ufw enable
```

## 📊 Monitoring

### Check Resource Usage:
```bash
# Memory usage
free -h

# Disk usage
df -h

# Docker container stats
docker stats

# Application logs
docker-compose logs fraudsight
```
