# WSL2 Streamlit Access Guide

## Problem Overview

When running Streamlit in WSL2, the Windows browser cannot connect to `http://localhost:8501` due to WSL2 networking architecture. WSL2 runs in a separate virtual network, and by default, Streamlit binds to `127.0.0.1` (localhost only), which is not accessible from Windows.

## Root Cause

1. **Streamlit default binding**: Streamlit binds to `127.0.0.1:8501` by default
2. **WSL2 network isolation**: WSL2 has its own IP address separate from Windows
3. **Windows cannot reach WSL localhost**: The Windows browser cannot directly access WSL's `127.0.0.1`

## Solutions (Ranked by Ease)

### Solution 1: Use the New WSL-Optimized Script (EASIEST)

We've created a WSL-specific launch script that automatically configures Streamlit for WSL2 networking.

**Steps:**

1. Stop the current Streamlit server (press `Ctrl+C` in the terminal where it's running)

2. Run the new WSL-optimized script:
   ```bash
   ./run_wsl.sh
   ```

3. The script will display both URLs to try:
   - `http://localhost:8501` (may work with VS Code port forwarding)
   - `http://<WSL-IP>:8501` (direct WSL IP access)

4. Open one of these URLs in your Windows browser

**Why this works:**
- Binds to `0.0.0.0` instead of `127.0.0.1` (all network interfaces)
- Disables CORS and XSRF protection for local development
- Provides both localhost and WSL IP addresses

---

### Solution 2: VS Code Port Forwarding (RECOMMENDED IF USING VS CODE)

If you're using VS Code connected to WSL, it can automatically forward ports.

**Steps:**

1. In VS Code, open the **Ports** panel:
   - Press `Ctrl+` ` (backtick) to open the terminal panel
   - Click on the **PORTS** tab (next to TERMINAL, PROBLEMS, etc.)

2. Check if port 8501 is listed:
   - If YES: Check the "Local Address" column - it should show `localhost:8501`
   - If NO: Click "Forward a Port" and enter `8501`

3. Right-click on port 8501 and ensure:
   - "Port Visibility" is set to "Public" or "Private"
   - The port is not paused

4. Try accessing `http://localhost:8501` in your Windows browser

**Why this works:**
- VS Code automatically creates a tunnel from Windows localhost to WSL
- No configuration changes needed
- Works seamlessly when properly configured

---

### Solution 3: Use WSL IP Address Directly

Access Streamlit using the WSL2 IP address instead of localhost.

**Steps:**

1. Get your WSL IP address:
   ```bash
   hostname -I | awk '{print $1}'
   ```
   Example output: `172.27.74.122`

2. In your Windows browser, navigate to:
   ```
   http://<WSL-IP>:8501
   ```
   Example: `http://172.27.74.122:8501`

3. Bookmark this URL for future use (note: WSL IP may change on restart)

**Why this works:**
- Directly accesses the WSL network interface
- Bypasses localhost routing issues
- Requires Streamlit to bind to `0.0.0.0` (configured in `.streamlit/config.toml`)

---

### Solution 4: Manual Streamlit Configuration

Configure Streamlit manually to work with WSL2.

**Steps:**

1. The configuration file has already been created at:
   `/home/arnavbansal/ExopatternNetV3/.streamlit/config.toml`

2. Stop the current Streamlit server (Ctrl+C)

3. Restart Streamlit using the original script:
   ```bash
   ./run.sh
   ```

4. Try accessing using either:
   - `http://localhost:8501` (if VS Code port forwarding is active)
   - `http://<WSL-IP>:8501` (get IP with `hostname -I`)

**Configuration details:**
```toml
[server]
enableCORS = false
enableXsrfProtection = false
headless = true
address = "0.0.0.0"  # Bind to all interfaces
port = 8501

[browser]
gatherUsageStats = false
serverAddress = "localhost"
serverPort = 8501
```

---

### Solution 5: Windows Firewall Configuration (If other solutions fail)

If none of the above work, Windows Firewall might be blocking the connection.

**Steps:**

1. Open Windows PowerShell as Administrator

2. Run this command to allow incoming connections on port 8501:
   ```powershell
   New-NetFirewallRule -DisplayName "WSL Streamlit" -Direction Inbound -LocalPort 8501 -Protocol TCP -Action Allow
   ```

3. Restart Streamlit in WSL

4. Try accessing `http://<WSL-IP>:8501`

---

## Quick Troubleshooting

### Check if Streamlit is running and listening:

```bash
ps aux | grep streamlit
ss -tlnp | grep 8501
```

### Verify WSL IP address:

```bash
hostname -I
```

### Test connection from within WSL:

```bash
curl http://localhost:8501
```

If this works but Windows browser doesn't connect, it's a networking/firewall issue.

### Check VS Code Port Forwarding:

1. Open VS Code Ports panel
2. Verify port 8501 is listed and forwarded
3. Check the "Local Address" matches what you're using in browser

---

## Recommended Workflow

**For VS Code Users:**
1. Use `./run_wsl.sh` to start the app
2. Ensure VS Code port forwarding is active (check Ports panel)
3. Access via `http://localhost:8501`

**For Terminal-Only Users:**
1. Use `./run_wsl.sh` to start the app
2. Note the WSL IP address shown in the output
3. Access via `http://<WSL-IP>:8501` in Windows browser
4. Bookmark the URL for convenience

---

## Files Modified/Created

1. **/.streamlit/config.toml** - Streamlit configuration for WSL2
2. **/run_wsl.sh** - WSL-optimized launch script
3. **WSL_SETUP_GUIDE.md** - This guide

## Additional Notes

- WSL IP address may change when Windows restarts
- The `run_wsl.sh` script displays the current WSL IP each time it runs
- For production deployments, consider using WSL1 or native Linux instead of WSL2
- VS Code port forwarding is the most convenient solution for development

---

## Still Having Issues?

1. Restart WSL:
   ```powershell
   # In Windows PowerShell
   wsl --shutdown
   ```
   Then reopen your WSL terminal

2. Check if another process is using port 8501:
   ```bash
   sudo lsof -i :8501
   ```

3. Try a different port:
   ```bash
   streamlit run frontend/app.py --server.port 8502
   ```
   Then access via `http://localhost:8502` or `http://<WSL-IP>:8502`
