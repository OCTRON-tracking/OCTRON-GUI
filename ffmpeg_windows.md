# Step-by-Step Tutorial for Installing FFmpeg on Windows

## Step 1: Download FFmpeg

1. Open your web browser and go to the official FFmpeg download page: [FFmpeg Download](https://ffmpeg.org/download.html).
2. Under the "Get packages & executable files" section, click on the Windows logo.
3. You will be redirected to a page with various builds. Click on the link for the "Windows builds from gyan.dev".
4. On the gyan.dev page, scroll down to the "Release builds" section and download the "ffmpeg-release-essentials.zip" file.

## Step 2: Extract the FFmpeg Zip File

1. Once the download is complete, navigate to the folder where the zip file was downloaded.
2. Right-click on the `ffmpeg-release-essentials.zip` file and select "Extract All...".
3. Choose a destination folder to extract the files to (e.g., `C:\ffmpeg`) and click "Extract".

## Step 3: Add FFmpeg to the System Path

1. Open the extracted folder (e.g., `C:\ffmpeg`) and navigate to the `bin` directory.
2. Copy the path to the `bin` directory (e.g., `C:\ffmpeg\bin`).

3. Open the Start menu, search for "Environment Variables", and select "Edit the system environment variables".
4. In the System Properties window, click on the "Environment Variables..." button.
5. In the Environment Variables window, find the "Path" variable under the "System variables" section and select it. Click "Edit...".
6. In the Edit Environment Variable window, click "New" and paste the path to the `bin` directory (e.g., `C:\ffmpeg\bin`). Click "OK" to close all windows.

## Step 4: Verify the Installation

1. Open the Command Prompt by pressing `Win + R`, typing `cmd`, and pressing `Enter`.
2. In the Command Prompt, type `ffmpeg -version` and press `Enter`.
3. If FFmpeg is installed correctly, you should see the version information for FFmpeg.

## Step 5: Use FFmpeg

You can now use FFmpeg from the Command Prompt or any other terminal on your Windows system.

## Troubleshooting

If you encounter any issues during the installation process, make sure to:
- Double-check that the path to the `bin` directory is correct.
- Ensure that the path is added to the "System variables" section, not the "User variables" section.
- Restart your Command Prompt or computer to apply the changes.
