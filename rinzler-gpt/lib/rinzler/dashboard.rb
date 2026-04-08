# frozen_string_literal: true

module Rinzler
  # Terminal TUI dashboard for the training loop.
  #
  # Uses VT100 scrolling regions to pin a fixed stats panel to the bottom of
  # the terminal while all freeform output (samples, checkpoints, log lines)
  # scrolls normally above it.
  #
  # Degrades gracefully: when stdout is not a TTY (piped to a log file, nohup,
  # etc.) the dashboard becomes a no-op and all output proceeds as plain text.
  #
  # Layout — 11 fixed lines at the bottom:
  #
  #   ╔══ rinzler-gpt | runs/8/ | 14:22:13 ══╗
  #   ║  Step  : 5200 / 100000  ████░  5.2%  ║
  #   ║  Elapsed: 2640s  ETA: ~12h 4m  0.51s ║
  #   ║  LR    : 2.98e-04                     ║
  #   ╠═══════════════════════════════════════╣
  #   ║  Train : 4.2513   Val : 4.9917        ║
  #   ║  Gap   : +0.74 (17.4%)               ║
  #   ║  Δgap  : -0.23 ↓    Norm: 0.847      ║
  #   ║  Status: ✓ OK                         ║
  #   ╚═══════════════════════════════════════╝
  #
  class Dashboard
    HEIGHT = 11   # lines reserved for the panel (including borders)

    # ANSI escape sequences
    CSI        = "\e["
    RESET      = "\e[0m"
    BOLD       = "\e[1m"
    RED        = "\e[31m"
    GREEN      = "\e[32m"
    YELLOW     = "\e[33m"
    CYAN       = "\e[36m"
    DIM        = "\e[2m"

    def initialize(total_steps:, start_step:, run_dir:, start_time:)
      @total_steps = total_steps
      @start_step  = start_step
      @run_dir     = File.basename(run_dir)
      @start_time  = start_time
      @tty         = $stdout.isatty

      # Current stats — populated by update()
      @step      = start_step
      @train     = nil
      @val       = nil
      @gap       = nil
      @gap_pct   = nil
      @trend     = nil
      @grad_norm = nil
      @lr        = nil
      @warning   = false
      @critical  = false
      @rows, @cols = terminal_size
    end

    # Activate the dashboard: set scroll region, draw initial frame.
    # Registers at_exit to guarantee teardown even on exception.
    def start!
      return unless @tty

      @rows, @cols = terminal_size
      set_scroll_region
      draw_frame
      at_exit { stop! }
    end

    # Deactivate: reset scroll region, move cursor below panel, final newline.
    def stop!
      return unless @tty

      # Reset scroll region to full terminal
      write "#{CSI}r"
      # Move below the panel
      write "#{CSI}#{@rows};1H\n"
    end

    # Redraw the stats rows with new data. Called every eval_every steps.
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

    # Emit a line of freeform output (sample text, checkpoint message, etc.)
    # into the scroll region. In no-TTY mode this is just puts.
    def emit(text = "")
      if @tty
        # Cursor is already positioned at the bottom of the scroll region
        # (maintained after each draw_frame). Just print — the scroll region
        # will scroll the text up naturally.
        $stdout.puts text
        $stdout.flush
      else
        puts text
      end
    end

    private

    # ── Drawing ──────────────────────────────────────────────────────────────────

    def draw_frame
      top = @rows - HEIGHT + 1   # first line of the panel

      save_cursor

      inner = @cols - 2   # usable width inside borders

      # Row 0: top border + title
      title     = " rinzler-gpt | #{@run_dir} | #{Time.now.strftime("%H:%M:%S")} "
      title_pad = inner - title.size
      goto(top, 1)
      write "#{CYAN}╔#{title}#{"═" * [title_pad, 0].max}╗#{RESET}"

      # Row 1: step + progress bar
      pct       = @total_steps > 0 ? @step.to_f / @total_steps : 0.0
      bar_width = [inner - 26, 10].max
      filled    = (pct * bar_width).round
      bar       = "█" * filled + "░" * (bar_width - filled)
      step_str  = "  Step : #{@step.to_s.rjust(6)} / #{@total_steps}"
      pct_str   = "#{(pct * 100).round(1).to_s.rjust(5)}%"
      goto(top + 1, 1)
      write "#{CYAN}║#{RESET}#{step_str}  #{CYAN}#{bar}#{RESET}  #{pct_str}#{" " * [inner - step_str.size - bar_width - pct_str.size - 3, 0].max}#{CYAN}║#{RESET}"

      # Row 2: elapsed + ETA + steps/sec
      elapsed   = @elapsed || (Time.now - @start_time).round(1)
      steps_done = [@step - @start_step, 1].max
      sps        = (steps_done / elapsed.to_f).round(2)
      remaining  = @total_steps - @step
      eta_str    = sps > 0 ? format_eta(remaining / sps) : "?"
      timing     = "  Elapsed: #{format_duration(elapsed)}   ETA: #{eta_str}   #{sps}s/step"
      goto(top + 2, 1)
      write "#{CYAN}║#{RESET}#{timing}#{" " * [inner - timing.size, 0].max}#{CYAN}║#{RESET}"

      # Row 3: LR
      lr_str = @lr ? "  LR    : #{format("%.4e", @lr)}" : "  LR    : —"
      goto(top + 3, 1)
      write "#{CYAN}║#{RESET}#{lr_str}#{" " * [inner - lr_str.size, 0].max}#{CYAN}║#{RESET}"

      # Row 4: divider
      goto(top + 4, 1)
      write "#{CYAN}╠#{"═" * inner}╣#{RESET}"

      # Row 5: train + val loss
      loss_str = @train ? "  Train : #{@train.to_s.ljust(8)}  Val : #{@val}" : "  Waiting for first eval..."
      goto(top + 5, 1)
      write "#{CYAN}║#{RESET}#{loss_str}#{" " * [inner - loss_str.size, 0].max}#{CYAN}║#{RESET}"

      # Row 6: gap
      if @gap
        gap_color = @warning ? (@critical ? RED : YELLOW) : GREEN
        gap_str   = "  Gap   : #{@gap >= 0 ? "+" : ""}#{@gap} (#{@gap_pct}%)"
        goto(top + 6, 1)
        write "#{CYAN}║#{RESET}#{gap_color}#{gap_str}#{RESET}#{" " * [inner - gap_str.size, 0].max}#{CYAN}║#{RESET}"
      else
        goto(top + 6, 1)
        write "#{CYAN}║#{" " * inner}║#{RESET}"
      end

      # Row 7: trend + grad norm
      trend_str = ""
      if @trend
        arrow       = @trend < 0 ? "↓" : (@trend > 0 ? "↑" : "→")
        trend_color = @trend < 0 ? GREEN : (@trend > 0.1 ? RED : YELLOW)
        trend_val   = "#{@trend >= 0 ? "+" : ""}#{@trend.round(4)}#{arrow}"
        trend_str   = "  Δgap  : #{trend_color}#{trend_val}#{RESET}"
        visible_len = "  Δgap  : #{trend_val}".size
      end
      norm_str = @grad_norm ? "  Norm: #{@grad_norm.round(3)}" : ""
      row7     = "#{trend_str}#{norm_str}"
      row7_vis = "#{trend_str.gsub(/\e\[[0-9;]*m/, "")}#{norm_str}"
      goto(top + 7, 1)
      write "#{CYAN}║#{RESET}#{row7}#{" " * [inner - row7_vis.size, 0].max}#{CYAN}║#{RESET}"

      # Row 8: status
      if @critical
        status = "#{RED}#{BOLD}✗ DIVERGING — gap exceeded critical threshold#{RESET}"
        status_vis = "✗ DIVERGING — gap exceeded critical threshold"
      elsif @warning
        status = "#{YELLOW}⚠  Generalization gap widening#{RESET}"
        status_vis = "⚠  Generalization gap widening"
      else
        status = "#{GREEN}✓ OK#{RESET}"
        status_vis = "✓ OK"
      end
      status_row = "  Status: #{status}"
      status_vis_row = "  Status: #{status_vis}"
      goto(top + 8, 1)
      write "#{CYAN}║#{RESET}#{status_row}#{" " * [inner - status_vis_row.size, 0].max}#{CYAN}║#{RESET}"

      # Row 9: bottom border
      goto(top + 9, 1)
      write "#{CYAN}╚#{"═" * inner}╝#{RESET}"

      restore_cursor
      $stdout.flush
    end

    # Plain text fallback for non-TTY — just prints a compact status line.
    def draw_plain
      return unless @train
      puts "step #{@step}/#{@total_steps} | train: #{@train} | val: #{@val} | gap: #{@gap >= 0 ? "+" : ""}#{@gap} (#{@gap_pct}%)"
    end

    # ── Terminal helpers ──────────────────────────────────────────────────────────

    def set_scroll_region
      scroll_rows = @rows - HEIGHT
      return if scroll_rows < 3   # terminal too small, skip

      # Set scrolling region to everything above the panel
      write "#{CSI}1;#{scroll_rows}r"
      # Move cursor to bottom of scroll region so output appears just above panel
      write "#{CSI}#{scroll_rows};1H"
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
