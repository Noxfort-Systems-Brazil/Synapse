#!/bin/bash
set -e

echo "[BUILD] Starting PyInstaller packaged build..."

# Create clean virtual environment
python3.12 -m venv /venv
source /venv/bin/activate

# Install required tools and dependencies
pip install --upgrade pip
pip install pyinstaller
pip install -r requirements.txt

# Run PyInstaller
echo "[BUILD] Running PyInstaller with synapse.spec..."
pyinstaller synapse.spec --clean --noconfirm

# Prepare deb package structure
echo "[BUILD] Preparing .deb structure..."
DEB_DIR="build_deb/synapse_1.0_amd64"
mkdir -p ${DEB_DIR}/opt/synapse
mkdir -p ${DEB_DIR}/usr/bin
mkdir -p ${DEB_DIR}/DEBIAN
mkdir -p ${DEB_DIR}/usr/share/applications
mkdir -p ${DEB_DIR}/usr/share/icons/hicolor/128x128/apps

# Copy Payload
cp -r dist/synapse/* ${DEB_DIR}/opt/synapse/

# Desktop File & Icon Integration
cp ui/assets/images/logo.png ${DEB_DIR}/usr/share/icons/hicolor/128x128/apps/synapse.png

cat << 'EOF' > ${DEB_DIR}/usr/share/applications/synapse.desktop
[Desktop Entry]
Name=SYNAPSE
Comment=A Gateway of Intelligent Perception for Traffic Management
Exec=/usr/bin/synapse
Icon=synapse
Terminal=false
Type=Application
Categories=Utility;
EOF
chmod 644 ${DEB_DIR}/usr/share/applications/synapse.desktop

# Startup Wrapper Script
cat << 'EOF' > ${DEB_DIR}/usr/bin/synapse
#!/bin/bash
export LD_LIBRARY_PATH=/opt/synapse:$LD_LIBRARY_PATH
exec /opt/synapse/synapse "$@"
EOF
chmod +x ${DEB_DIR}/usr/bin/synapse

# DEB metadata file
cat << EOF > ${DEB_DIR}/DEBIAN/control
Package: synapse
Version: 1.0
Section: utils
Priority: optional
Architecture: amd64
Maintainer: Noxfort Systems <suporte@noxfort.com>
Description: SYNAPSE - Intelligent Perception Gateway
 Self-contained bundle including dependencies.
EOF

# Build package
echo "[BUILD] Packaging into .deb file..."
dpkg-deb --build ${DEB_DIR}
mv build_deb/synapse_1.0_amd64.deb .

echo "[BUILD] Success! File synapse_1.0_amd64.deb generated."
