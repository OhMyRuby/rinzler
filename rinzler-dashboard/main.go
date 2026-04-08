package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"strings"
	"time"

	"github.com/charmbracelet/bubbles/progress"
	"github.com/charmbracelet/bubbles/viewport"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
)

// ── JSON protocol ─────────────────────────────────────────────────────────────
//
// Each line from the trainer is one of these event types.

type Event struct {
	Type string `json:"type"`

	// eval
	Step      int     `json:"step"`
	Total     int     `json:"total"`
	Train     float64 `json:"train"`
	Val       float64 `json:"val"`
	Gap       float64 `json:"gap"`
	GapPct    float64 `json:"gap_pct"`
	Trend     *float64 `json:"trend"`
	LR        float64 `json:"lr"`
	GradNorm  float64 `json:"grad_norm"`
	Elapsed   float64 `json:"elapsed"`
	Warning   bool    `json:"warning"`
	Critical  bool    `json:"critical"`

	// sample / checkpoint / message
	Text    string `json:"text"`
	Path    string `json:"path"`
	Message string `json:"message"`
}

// ── Bubble Tea messages ───────────────────────────────────────────────────────

type eventMsg Event
type doneMsg struct{}
type errMsg struct{ err error }

// ── Styles ────────────────────────────────────────────────────────────────────

var (
	styleBorder   = lipgloss.NewStyle().Border(lipgloss.RoundedBorder()).BorderForeground(lipgloss.Color("6"))
	styleLabel    = lipgloss.NewStyle().Foreground(lipgloss.Color("8"))
	styleValue    = lipgloss.NewStyle().Foreground(lipgloss.Color("15"))
	styleOK       = lipgloss.NewStyle().Foreground(lipgloss.Color("2"))
	styleWarn     = lipgloss.NewStyle().Foreground(lipgloss.Color("11"))
	styleCrit     = lipgloss.NewStyle().Foreground(lipgloss.Color("9")).Bold(true)
	styleDim      = lipgloss.NewStyle().Foreground(lipgloss.Color("8"))
	styleInfo     = lipgloss.NewStyle().Foreground(lipgloss.Color("12"))
	styleSampleHdr = lipgloss.NewStyle().Foreground(lipgloss.Color("5")).Bold(true)
	styleSampleBox = lipgloss.NewStyle().
			Border(lipgloss.RoundedBorder()).
			BorderForeground(lipgloss.Color("5")).
			Padding(0, 1)
	styleCheckpoint = lipgloss.NewStyle().Foreground(lipgloss.Color("8"))
	styleWarnBox    = lipgloss.NewStyle().
			Border(lipgloss.NormalBorder()).
			BorderForeground(lipgloss.Color("11")).
			Foreground(lipgloss.Color("11")).
			Padding(0, 1)
	styleCritBox = lipgloss.NewStyle().
			Border(lipgloss.DoubleBorder()).
			BorderForeground(lipgloss.Color("9")).
			Foreground(lipgloss.Color("9")).
			Bold(true).
			Padding(0, 1)
)

// ── Model ─────────────────────────────────────────────────────────────────────

type model struct {
	width  int
	height int

	// latest eval stats
	step     int
	total    int
	train    *float64
	val      *float64
	gap      *float64
	gapPct   *float64
	trend    *float64
	lr       *float64
	gradNorm *float64
	elapsed  float64
	warning  bool
	critical bool
	runDir   string

	// rolling step-rate tracking
	prevStep    int
	prevElapsed float64
	stepRates   []float64 // per-interval s/step, capped at rateWindow


	progress progress.Model
	vp       viewport.Model
	logLines []string
	ready    bool
	done     bool
}

func newModel(runDir string) model {
	p := progress.New(
		progress.WithSolidFill("6"),
		progress.WithoutPercentage(),
	)
	return model{
		runDir:   runDir,
		progress: p,
	}
}

func (m model) Init() tea.Cmd {
	return nil
}

func (m model) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {

	case tea.WindowSizeMsg:
		m.width = msg.Width
		m.height = msg.Height
		m.progress.Width = m.width - 4 // panel inner width minus padding
		if !m.ready {
			m.vp = viewport.New(m.width, m.logHeight())
			m.vp.SetContent("")
			m.ready = true
		} else {
			m.vp.Width = m.width
			m.vp.Height = m.logHeight()
		}
		return m, nil

	case eventMsg:
		e := Event(msg)
		switch e.Type {
		case "eval":
			// Per-interval step rate (s/step over the last eval window).
			if m.prevStep > 0 && e.Step > m.prevStep {
				dt := e.Elapsed - m.prevElapsed
				ds := float64(e.Step - m.prevStep)
				if dt > 0 && ds > 0 {
					const rateWindow = 60
					m.stepRates = append(m.stepRates, dt/ds)
					if len(m.stepRates) > rateWindow {
						m.stepRates = m.stepRates[len(m.stepRates)-rateWindow:]
					}
				}
			}
			m.prevStep = e.Step
			m.prevElapsed = e.Elapsed

			m.step = e.Step
			m.total = e.Total
			m.train = &e.Train
			m.val = &e.Val
			m.gap = &e.Gap
			m.gapPct = &e.GapPct
			m.trend = e.Trend
			m.lr = &e.LR
			m.gradNorm = &e.GradNorm
			m.elapsed = e.Elapsed
			m.warning = e.Warning
			m.critical = e.Critical

		case "sample":
			hdr := styleSampleHdr.Render(fmt.Sprintf("─── step %d · temperature=0.8 ───", e.Step))
			box := styleSampleBox.Width(m.width - 4).Render(e.Text)
			m.appendLog(hdr + "\n" + box)

		case "checkpoint":
			m.appendLog(styleCheckpoint.Render("✓ checkpoint  " + e.Path))

		case "warn":
			m.appendLog(styleWarnBox.Width(m.width - 4).Render("⚠  " + e.Message))

		case "critical":
			m.appendLog(styleCritBox.Width(m.width - 4).Render("✗  " + e.Message))

		case "log":
			m.appendLog(styleDim.Render(e.Message))

		case "done":
			m.done = true
			m.appendLog(styleOK.Render(fmt.Sprintf("✓ Training complete in %s.", formatDuration(e.Elapsed))))
		}
		return m, nil

	case doneMsg:
		return m, tea.Quit

	case errMsg:
		return m, tea.Quit

	case tea.KeyMsg:
		if msg.String() == "q" || msg.String() == "ctrl+c" {
			return m, tea.Quit
		}
	}

	// Forward scroll keys to viewport
	var cmd tea.Cmd
	m.vp, cmd = m.vp.Update(msg)
	return m, cmd
}

func (m *model) appendLog(line string) {
	m.logLines = append(m.logLines, line)
	m.vp.SetContent(strings.Join(m.logLines, "\n"))
	m.vp.GotoBottom()
}

func (m model) View() string {
	if !m.ready || m.width == 0 {
		return ""
	}

	panel := m.renderPanel()
	log := m.vp.View()
	return panel + "\n" + log
}

func (m model) logHeight() int {
	// Panel height: border(2) + step(1) + progress(1) + timing(1) + lr(1) + divider(1) + losses(1) + gap(1) + trend(1) + status(1)
	panelHeight := 11
	h := m.height - panelHeight - 1 // -1 for the newline between panel and log
	if h < 3 {
		h = 3
	}
	return h
}

func (m model) renderPanel() string {
	inner := m.width - 2 // inside border chars (lipgloss border adds 1 per side)

	// ── Progress bar row
	pct := 0.0
	if m.total > 0 {
		pct = float64(m.step) / float64(m.total)
	}
	stepStr := fmt.Sprintf("  Step : %6d / %d", m.step, m.total)
	pctStr := fmt.Sprintf("%5.1f%%", pct*100)
	barWidth := inner - len(stepStr) - len(pctStr) - 4
	if barWidth < 4 {
		barWidth = 4
	}
	m.progress.Width = barWidth
	barStr := m.progress.ViewAs(pct)
	progressRow := stepStr + "  " + barStr + "  " + pctStr

	// ── Timing row — rolling avg rate + braille sparkline
	sps := 0.0
	if m.elapsed > 0 && m.step > 0 {
		sps = m.elapsed / float64(m.step) // global avg fallback
	}
	avgSps := sps
	if len(m.stepRates) > 0 {
		avgSps = rollingAvg(m.stepRates, 20) // 20-eval rolling avg
	}
	remaining := float64(m.total - m.step)
	etaStr := "?"
	if avgSps > 0 {
		etaStr = formatDuration(remaining * avgSps)
	}
	sparkWidth := 20
	spark := brailleSpark(m.stepRates, sparkWidth)
	timingRow := fmt.Sprintf("  Elapsed: %s   ETA: %s   %.2fs/step  %s",
		formatDuration(m.elapsed), etaStr, avgSps, spark)

	// ── LR row
	lrRow := "  LR    : —"
	if m.lr != nil {
		lrRow = fmt.Sprintf("  LR    : %.4e", *m.lr)
	}

	// ── Loss row
	lossRow := "  Waiting for first eval..."
	if m.train != nil {
		lossRow = fmt.Sprintf("  Train : %-8s  Val : %s",
			fmt.Sprintf("%.4f", *m.train),
			fmt.Sprintf("%.4f", *m.val))
	}

	// ── Gap row
	gapRow := ""
	if m.gap != nil {
		sign := "+"
		if *m.gap < 0 {
			sign = ""
		}
		gapStr := fmt.Sprintf("  Gap   : %s%.4f (%.1f%%)", sign, *m.gap, *m.gapPct)
		if m.critical {
			gapRow = styleCrit.Render(gapStr)
		} else if m.warning {
			gapRow = styleWarn.Render(gapStr)
		} else {
			gapRow = styleOK.Render(gapStr)
		}
	}

	// ── Trend + norm row
	trendRow := ""
	if m.trend != nil {
		arrow := "→"
		if *m.trend < 0 {
			arrow = "↓"
		} else if *m.trend > 0 {
			arrow = "↑"
		}
		trendVal := fmt.Sprintf("%+.4f%s", *m.trend, arrow)
		var trendStyled string
		if *m.trend < 0 {
			trendStyled = styleOK.Render(trendVal)
		} else if *m.trend > 0.1 {
			trendStyled = styleCrit.Render(trendVal)
		} else {
			trendStyled = styleWarn.Render(trendVal)
		}
		trendRow = "  Δgap  : " + trendStyled
	}
	if m.gradNorm != nil {
		sep := ""
		if trendRow != "" {
			sep = "   "
		} else {
			sep = "  "
		}
		trendRow += sep + fmt.Sprintf("Norm: %.3f", *m.gradNorm)
	}

	// ── Status row
	var statusRow string
	if m.critical {
		statusRow = "  Status: " + styleCrit.Render("✗ DIVERGING — gap exceeded critical threshold")
	} else if m.warning {
		statusRow = "  Status: " + styleWarn.Render("⚠  Generalization gap widening")
	} else if m.train != nil {
		statusRow = "  Status: " + styleOK.Render("✓ OK")
	} else {
		statusRow = "  Status: " + styleDim.Render("initializing...")
	}

	divider := strings.Repeat("─", inner)

	title := fmt.Sprintf(" rinzler-gpt | %s | %s ", m.runDir, time.Now().Format("15:04:05"))

	content := strings.Join([]string{
		title,
		progressRow,
		timingRow,
		lrRow,
		divider,
		lossRow,
		gapRow,
		trendRow,
		statusRow,
	}, "\n")

	return styleBorder.Width(inner).Render(content)
}

// ── stdin reader ──────────────────────────────────────────────────────────────
//
// Runs in a goroutine for the lifetime of the program. Sends parsed events to
// the Bubble Tea program via p.Send(). A single scanner is reused — creating a
// new one each call would lose buffered bytes between reads.

func readStdinLoop(p *tea.Program) {
	scanner := bufio.NewScanner(os.Stdin)
	for scanner.Scan() {
		var e Event
		if err := json.Unmarshal(scanner.Bytes(), &e); err != nil {
			p.Send(eventMsg{Type: "log", Message: scanner.Text()})
			continue
		}
		p.Send(eventMsg(e))
		if e.Type == "done" {
			break
		}
	}
	if err := scanner.Err(); err != nil {
		p.Send(errMsg{err})
	}
	p.Send(doneMsg{})
}

// ── Helpers ───────────────────────────────────────────────────────────────────

func rollingAvg(values []float64, window int) float64 {
	if len(values) == 0 {
		return 0
	}
	start := len(values) - window
	if start < 0 {
		start = 0
	}
	sum := 0.0
	for _, v := range values[start:] {
		sum += v
	}
	return sum / float64(len(values[start:]))
}

// brailleSpark renders values as a compact braille sparkline.
// Each braille character encodes two data points side by side (2 cols wide, 4 levels tall).
// Returns a string of `width` braille characters covering the most recent width*2 values.
func brailleSpark(values []float64, width int) string {
	if len(values) == 0 {
		return strings.Repeat("⠀", width)
	}

	// Take the most recent width*2 samples.
	capacity := width * 2
	vals := make([]float64, capacity)
	src := values
	if len(src) > capacity {
		src = src[len(src)-capacity:]
	}
	// Right-align: pad left with zeros if not enough data yet.
	copy(vals[capacity-len(src):], src)

	min, max := math.MaxFloat64, -math.MaxFloat64
	for _, v := range src {
		if v < min {
			min = v
		}
		if v > max {
			max = v
		}
	}
	if max <= min {
		max = min + 1
	}

	// Map a value to 0-4 height levels.
	level := func(v float64) int {
		l := int((v - min) / (max - min) * 4.0)
		if l > 4 {
			l = 4
		}
		if l < 0 {
			l = 0
		}
		return l
	}

	// Braille bit patterns for left and right column bars (bottom-to-top).
	// Left column uses dots 7,3,2,1 (bits 6,2,1,0); right uses dots 8,6,5,4 (bits 7,5,4,3).
	leftBits  := [5]rune{0x00, 0x40, 0x44, 0x46, 0x47}
	rightBits := [5]rune{0x00, 0x80, 0xa0, 0xb0, 0xb8}

	var sb strings.Builder
	for i := 0; i < capacity; i += 2 {
		l := level(vals[i])
		r := level(vals[i+1])
		sb.WriteRune(0x2800 + leftBits[l] + rightBits[r])
	}
	return sb.String()
}

func formatDuration(seconds float64) string {
	if math.IsInf(seconds, 0) || math.IsNaN(seconds) || seconds < 0 {
		return "?"
	}
	h := int(seconds) / 3600
	m := (int(seconds) % 3600) / 60
	s := int(seconds) % 60
	if h > 0 {
		return fmt.Sprintf("~%dh %dm", h, m)
	}
	if m > 0 {
		return fmt.Sprintf("~%dm %ds", m, s)
	}
	return fmt.Sprintf("%ds", s)
}

// ── Main ──────────────────────────────────────────────────────────────────────

func main() {
	runDir := "."
	if len(os.Args) > 1 {
		runDir = os.Args[1]
	}

	// Open /dev/tty for keyboard input so Bubble Tea doesn't compete with
	// our JSONL reader on os.Stdin.
	tty, err := os.Open("/dev/tty")
	if err != nil {
		fmt.Fprintln(os.Stderr, "could not open /dev/tty:", err)
		os.Exit(1)
	}
	defer tty.Close()

	p := tea.NewProgram(
		newModel(runDir),
		tea.WithAltScreen(),
		tea.WithInput(tty),
	)

	// Read JSONL from stdin in a goroutine; send parsed events to the program.
	go readStdinLoop(p)

	if _, err := p.Run(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
