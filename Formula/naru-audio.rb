class NaruAudio < Formula
  desc "Local speech-to-text and text-to-speech daemon for Naru"
  homepage "https://github.com/simonspoon/naru-audio"
  version "0.1.0"
  license "GPL-3.0-or-later"

  on_macos do
    if Hardware::CPU.arm?
      url "https://github.com/simonspoon/naru-audio/releases/download/v0.1.0/naru-audio-darwin-arm64"
      sha256 "1111111111111111111111111111111111111111111111111111111111111111"
    else
      url "https://github.com/simonspoon/naru-audio/releases/download/v0.1.0/naru-audio-darwin-amd64"
      sha256 "2222222222222222222222222222222222222222222222222222222222222222"
    end
  end

  on_linux do
    if Hardware::CPU.arm?
      url "https://github.com/simonspoon/naru-audio/releases/download/v0.1.0/naru-audio-linux-arm64"
      sha256 "3333333333333333333333333333333333333333333333333333333333333333"
    else
      url "https://github.com/simonspoon/naru-audio/releases/download/v0.1.0/naru-audio-linux-amd64"
      sha256 "4444444444444444444444444444444444444444444444444444444444444444"
    end
  end

  def install
    binary = Dir["naru-audio-*"].first || "naru-audio"
    bin.install binary => "naru-audio"
  end

  def caveats
    "naru-audio pull default && brew services start naru-audio"
  end

  service do
    run [opt_bin/"naru-audio", "serve"]
    keep_alive crashed: true
    process_type :interactive # audio latency; avoid background throttling
    log_path var/"log/naru-audio.log"
    error_log_path var/"log/naru-audio.log"
    environment_variables RUST_LOG: "naru_audio=info"
  end

  test do
    assert_match "Local STT/TTS daemon for Naru", shell_output("#{bin}/naru-audio --help")
  end
end
