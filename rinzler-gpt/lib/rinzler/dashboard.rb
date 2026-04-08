# frozen_string_literal: true

require "shellwords"

module Rinzler
  # Terminal TUI dashboard for the training loop.
  #
  # Uses VT100 scrolling regions to pin a fixed stats panel to the TOP of
  # the terminal while all freeform output (samples, checkpoints, log lines)
  # scrolls normally below it.
  #
  # Degrades gracefully: when stdout is not a TTY (piped to a log file, nohup,
  # etc.) the dashboard becomes a no-op and all output proceeds as plain text.
  #
  # Layout — 11 fixed lines at the top:
  #
  #   ╔══ rinzler-gpt | runs/8/ | 14:22:13 ══════════════════════════════════╗
  #   ║  Step :   5200 / 100000  ████████████████████████░░░░░░░░░░░  5.2%  ║
  #   ║  Elapsed: 44m 0s   ETA: ~12h 4m   0.51s/step                        ║
  #   ║  LR    : 2.9800e-04                                                  ║
  #   ╠══════════════════════════════════════════════════════════════════════╣
  #   ║  Train : 4.2513   Val : 4.9917                                       ║
  #   ║  Gap   : +0.7404 (17.4%)                                             ║
  #   ║  Δgap  : -0.2300↓   Norm: 0.847                                      ║
  #   ║  Status: ✓ OK                                                        ║
  #   ╚══════════════════════════════════════════════════════════════════════╝
  #                      ↑ panel pinned here
  #   [log output scrolls below]
  #
  class Dashboard
    HEIGHT = 10   # lines reserved for the panel (including borders)

    # ANSI escape sequences
    CSI   = "\e["
    RESET = "\e[0m"
    BOLD  = "\e[1m"
    RED   = "\e[31m"
    GREEN = "\e[32m"
    YELLOW = "\e[33m"
    CYAN  = "\e[36m"
    DIM   = "\e[2m"

    def initialize(total_steps:, start_step:, run_dir:, start_time:)
      @total_steps = total_steps
      @start_step  = start_step
      @run_dir     = File.basename(run_dir)
      @start_time  = start_time
      @tty         = $stdout.isatty
      @gum         = @tty && system("which gum > /dev/null 2>&1")

      @step      = start_step
      @train     = nil
      @val       = nil
      @gap       = nil
      @gap_pct   = nil
      @trend     = nil
      @grad_norm = nil
      @lr        = nil
      @elapsed   = nil
      @warning   = false
      @critical  = false
      @rows, @cols = terminal_size
    end

    # Activate the dashboard: clear screen, set scroll region below panel, draw frame.
    def start!
      return unless @tty

      @rows, @cols = terminal_size
      write "\e[2J\e[H"   # clear screen, cursor to home
      draw_frame
      set_scroll_region
      at_exit { stop! }

      # Reflow on terminal resize
      Signal.trap("WINCH") do
        @rows, @cols = terminal_size
        draw_frame
        set_scroll_region
      end
    end

    # Deactivate: reset scroll region, move cursor to bottom for clean exit output.
    def stop!
      return unless @tty

      Signal.trap("WINCH", "DEFAULT")
      write "#{CSI}r"              # reset scroll region to full terminal
      write "#{CSI}#{@rows};1H\n" # cursor to last row
      $stdout.flush
    end

    # Redraw the stats panel with new data. Called every eval_every steps.
    def update(step:, train:, val:, gap:, gap_pct:, trend:, lr:, grad_norm:, elapsed:, warning:, critical: false)
      @step      = step
      @train     = train
      @val       = val
      @gap       = gap
      @gap_pct   = gap_pct
      @trend     = trend
      @lr        = lr
      @grad_norm = grad_norm
      @elapsed   = elapsed
      @warning   = warning
      @critical  = critical

      return draw_plain unless @tty

      @rows, @cols = terminal_size
      draw_frame
    end

    # Emit freeform output into the scroll region below the panel.
    # Accepts an optional :style keyword to select gum styling:
    #   :sample   — bordered box for generated text samples
    #   :warn     — yellow warning callout
    #   :critical — red critical callout
    #   :info     — dim muted line (checkpoints, saves)
    # In no-TTY mode this is just puts.
    def emit(text = "", style: nil)
      if @tty && @gum && style
        gum_style(text, style)   # writes directly to stdout via system()
      elsif @tty
        $stdout.puts text
        $stdout.flush
      else
        puts text
      end
    end

    private

    # ── Drawing ──────────────────────────────────────────────────────────────────

    def draw_frame
      save_cursor

      inner = @cols - 2   # usable width inside the box borders

      # Row 0: top border + title
      title     = " rinzler-gpt | #{@run_dir} | #{Time.now.strftime("%H:%M:%S")} "
      title_pad = [inner - title.size, 0].max
      goto(1, 1)
      write "#{CYAN}╔#{title}#{"═" * title_pad}╗#{RESET}"

      # Row 1: step + progress bar
      pct      = @total_steps > 0 ? @step.to_f / @total_steps : 0.0
      step_str = "  Step : #{@step.to_s.rjust(6)} / #{@total_steps}"
      pct_str  = "#{(pct * 100).round(1).to_s.rjust(5)}%"
      bar_width = [inner - step_str.size - pct_str.size - 4, 4].max
      filled    = (pct * bar_width).round
      bar       = "█" * filled + "░" * (bar_width - filled)
      # content = step_str(plain) + "  " + bar(plain) + "  " + pct_str(plain)
      # total visible = step_str.size + 2 + bar_width + 2 + pct_str.size; pad remainder
      row1_content = "#{step_str}  #{CYAN}#{bar}#{RESET}  #{pct_str}"
      goto(2, 1)
      write "#{CYAN}║#{RESET}#{row1_content}#{" " * [inner - step_str.size - bar_width - pct_str.size - 4, 0].max}#{CYAN}║#{RESET}"

      # Row 2: elapsed + ETA + steps/sec
      elapsed    = @elapsed || (Time.now - @start_time).round(1)
      steps_done = [@step - @start_step, 1].max
      sps        = (steps_done / elapsed.to_f).round(2)
      remaining  = @total_steps - @step
      eta_str    = sps > 0 ? format_eta(remaining / sps) : "?"
      timing     = "  Elapsed: #{format_duration(elapsed)}   ETA: #{eta_str}   #{sps}s/step"
      goto(3, 1)
      write box_row(timing, inner)

      # Row 3: LR
      lr_str = @lr ? "  LR    : #{format("%.4e", @lr)}" : "  LR    : —"
      goto(4, 1)
      write box_row(lr_str, inner)

      # Row 4: divider
      goto(5, 1)
      write "#{CYAN}╠#{"═" * inner}╣#{RESET}"

      # Row 5: train + val loss
      loss_str = @train ? "  Train : #{@train.to_s.ljust(8)}  Val : #{@val}" : "  Waiting for first eval..."
      goto(6, 1)
      write box_row(loss_str, inner)

      # Row 6: gap (colored — use vis_size for padding)
      goto(7, 1)
      if @gap
        gap_color   = @warning ? (@critical ? RED : YELLOW) : GREEN
        gap_plain   = "  Gap   : #{@gap >= 0 ? "+" : ""}#{@gap} (#{@gap_pct}%)"
        gap_styled  = "#{gap_color}#{gap_plain}#{RESET}"
        write "#{CYAN}║#{RESET}#{gap_styled}#{" " * [inner - gap_plain.size, 0].max}#{CYAN}║#{RESET}"
      else
        write "#{CYAN}║#{" " * inner}║#{RESET}"
      end

      # Row 7: trend + grad norm (colored trend — track plain separately)
      goto(8, 1)
      trend_plain = ""
      trend_styled = ""
      if @trend
        arrow        = @trend < 0 ? "↓" : (@trend > 0 ? "↑" : "→")
        trend_color  = @trend < 0 ? GREEN : (@trend > 0.1 ? RED : YELLOW)
        trend_val    = "#{@trend >= 0 ? "+" : ""}#{@trend.round(4)}#{arrow}"
        trend_plain  = "  Δgap  : #{trend_val}"
        trend_styled = "  Δgap  : #{trend_color}#{trend_val}#{RESET}"
      end
      norm_plain  = @grad_norm ? "  Norm: #{@grad_norm.round(3)}" : ""
      row7_plain  = "#{trend_plain}#{norm_plain}"
      row7_styled = "#{trend_styled}#{norm_plain}"
      write "#{CYAN}║#{RESET}#{row7_styled}#{" " * [inner - row7_plain.size, 0].max}#{CYAN}║#{RESET}"

      # Row 8: status (colored — track plain separately)
      goto(9, 1)
      if @critical
        status_plain  = "  Status: ✗ DIVERGING — gap exceeded critical threshold"
        status_styled = "  Status: #{RED}#{BOLD}✗ DIVERGING — gap exceeded critical threshold#{RESET}"
      elsif @warning
        status_plain  = "  Status: ⚠  Generalization gap widening"
        status_styled = "  Status: #{YELLOW}⚠  Generalization gap widening#{RESET}"
      else
        status_plain  = "  Status: ✓ OK"
        status_styled = "  Status: #{GREEN}✓ OK#{RESET}"
      end
      write "#{CYAN}║#{RESET}#{status_styled}#{" " * [inner - status_plain.size, 0].max}#{CYAN}║#{RESET}"

      # Row 9: bottom border
      goto(10, 1)
      write "#{CYAN}╚#{"═" * inner}╝#{RESET}"

      restore_cursor
      $stdout.flush
    end

    # Render a plain-text content string as a full-width box row.
    # Content must contain no ANSI codes — padding derived from content.size directly.
    def box_row(content, inner)
      "#{CYAN}║#{RESET}#{content}#{" " * [inner - content.size, 0].max}#{CYAN}║#{RESET}"
    end

    # Plain text fallback for non-TTY — compact status line via gum log.
    def draw_plain
      return unless @train
      line = "step #{@step}/#{@total_steps} | train: #{@train} | val: #{@val} | gap: #{@gap >= 0 ? "+" : ""}#{@gap} (#{@gap_pct}%)"
      if @gum
        level = @critical ? "error" : (@warning ? "warn" : "info")
        system("gum", "log", "--level", level, line)
      else
        puts line
      end
    end

    # Render styled output via gum, writing directly to stdout.
    # Uses array-form system() so text with newlines is passed as a single arg
    # without shell interpretation.
    def gum_style(text, style)
      w = (@cols - 4).to_s
      case style
      when :sample
        system("gum", "style",
               "--border", "rounded", "--border-foreground", "99",
               "--padding", "0 1", "--width", w, text)
      when :warn
        system("gum", "style",
               "--border", "normal", "--border-foreground", "214",
               "--foreground", "214", "--padding", "0 1", text)
      when :critical
        system("gum", "style",
               "--border", "double", "--border-foreground", "196",
               "--foreground", "196", "--bold", "--padding", "0 1", text)
      when :info
        system("gum", "style", "--foreground", "240", text)
      end
      $stdout.flush
    end

    # ── Terminal helpers ──────────────────────────────────────────────────────────

    def set_scroll_region
      scroll_start = HEIGHT + 1
      return if @rows - HEIGHT < 3

      write "#{CSI}#{scroll_start};#{@rows}r"   # scroll region: below panel → bottom
      write "#{CSI}#{@rows};1H"                  # cursor to bottom of scroll area
      $stdout.flush
    end

    def save_cursor    = write("\e[s")
    def restore_cursor = write("\e[u")
    def goto(row, col) = write("#{CSI}#{row};#{col}H")
    def write(str)     = $stdout.write(str)

    def terminal_size
      $stdout.winsize
    rescue Errno::ENOTTY, NoMethodError
      [24, 80]
    end

    def format_eta(seconds)
      return "?" if seconds < 0 || seconds.infinite? rescue return "?"
      h = (seconds / 3600).to_i
      m = ((seconds % 3600) / 60).to_i
      h > 0 ? "~#{h}h #{m}m" : "~#{m}m"
    end

    def format_duration(seconds)
      h = (seconds / 3600).to_i
      m = ((seconds % 3600) / 60).to_i
      s = (seconds % 60).to_i
      h > 0 ? "#{h}h #{m}m #{s}s" : "#{m}m #{s}s"
    end
  end
end
