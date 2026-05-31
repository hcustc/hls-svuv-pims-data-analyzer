# Docker 部署方案

这是一套适合阿里云轻量应用服务器（容器环境）或普通 Linux 主机的部署方式。
默认采用“同源部署”：Nginx 只负责 HTTPS 和反向代理，FastAPI 同时提供网页和 API。

## 1. 目录结构

建议把仓库放到如下目录：

```text
/srv/bl03u/
```

仓库根目录中已经包含：

```text
Dockerfile
docker-compose.yml
.env.example
deploy/nginx/bl03u.conf
```

## 2. 准备服务器

安装 Docker 和 Docker Compose 后，把仓库同步到服务器：

```bash
git clone <your-repo-url> /srv/bl03u
cd /srv/bl03u
cp .env.example .env
```

编辑 `.env`，把 `BL03U_ADMIN_TOKEN` 改成强随机字符串。

## 3. 构建并启动

```bash
docker compose up -d --build
docker compose ps
```

容器会监听在 `127.0.0.1:8000`，不会直接暴露到公网。

## 4. Nginx 反向代理

把 `deploy/nginx/bl03u.conf` 复制到 `/etc/nginx/conf.d/bl03u.conf` 或站点目录后重载 Nginx。

```bash
sudo nginx -t
sudo systemctl reload nginx
```

如果你要启用 HTTPS，再用 certbot 为域名申请证书。

### 4.1 只用公网 IP 的部署方式

如果你暂时不申请域名，公网 IP 方案也能用，适合先跑通服务。

推荐访问地址：

```text
http://<你的公网IP>/
```

这样做的关键点是：

- Nginx 监听 `80` 端口，并把请求转发到本机的 `127.0.0.1:8000`
- 前端和后端共用同一个入口，不会出现 GitHub Pages 那种跨域问题
- 以后如果补域名，只要把 `server_name` 改成域名并加 HTTPS 即可

仓库里的 `deploy/nginx/bl03u.conf` 已经使用 `server_name _;`，可以直接匹配公网 IP 访问。

注意：

- 纯公网 IP 场景一般先用 `http://` 测试
- 如果你以后把前端放在 GitHub Pages，再去请求这个 `http://` 公网 IP，浏览器会拦截混合内容
- 长期对外开放时，域名 + HTTPS 仍然是更稳妥的方案

## 5. 访问方式

### 方案 A：同源部署

浏览器直接访问你的域名，例如：

```text
https://api.example.com/
```

此时网页和 API 是同一个来源，不需要额外配置前端 API 地址。

### 方案 B：GitHub Pages 前端 + 独立后端

如果前端仍放在 GitHub Pages，访问时给前端加上 `api` 参数：

```text
https://username.github.io/your-repo/?api=https://api.example.com/api
```

此时需要保留当前后端的 CORS 设置，或者把 `allow_origins` 收紧为你的 GitHub Pages 域名。

### 方案 C：公网 IP + 同源前后端

这是你当前更推荐的路径：前端和后端都部署在阿里云主机上，通过公网 IP 或后续域名统一访问。

访问方式：

```text
http://<你的公网IP>/
```

这种方式不需要额外配置前端 API 地址，因为前端和 API 是同源的。

## 6. 数据与持久化

- `./database` 会映射到容器内的 `/app/database`，用于保存维护库与备份。
- `./data` 会映射到容器内的 `/data/bl03u`，用于你允许后端读取的数据目录。
- 建议定期备份 `database/` 目录。

## 7. 常用命令

```bash
docker compose logs -f bl03u
docker compose restart bl03u
docker compose down
```

## 8. 适用场景

- 个人站点
- 实验室内部服务
- 小规模线上访问
- 需要后续迁移到阿里云 ECS 或容器平台的项目

如果你要的是生产级公网部署，建议再加：

- 域名
- HTTPS
- 强随机管理员 token
- CORS 白名单
- 定期备份

## 9. 开机自启

如果你希望服务器重启后自动恢复服务，可以把仓库里的 `deploy/systemd/bl03u-docker.service` 放到 `/etc/systemd/system/bl03u-docker.service`，然后执行：

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now bl03u-docker.service
```

这个服务会在启动时调用 `docker compose up -d --build`，停止时调用 `docker compose down`。