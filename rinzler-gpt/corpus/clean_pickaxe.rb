#!/usr/bin/env ruby
# frozen_string_literal: true
#
# clean_pickaxe.rb — strip PDF artifacts from pickaxe6.txt and restore structure
#
# Removes:
#   - Page headers:  "Chapter N. Title • N"
#   - Chapter stamps: "CHAPTER N" on a line by itself
#   - Footers:       "report erratum • discuss" (standalone lines AND fused mid-line)
#   - Special chars: ß (PDF beta marker)
#
# Fixes:
#   - Hyphenated line breaks: "word-\ncontinuation" → "wordcontinuation"
#   - Paragraph spacing: ensures blank line between paragraphs
#   - Heading spacing: blank lines before/after section headings

input  = File.join(__dir__, "pickaxe6.txt")
output = File.join(__dir__, "pickaxe6_clean.txt")

lines = File.readlines(input, chomp: true)

cleaned = []
i       = 0

while i < lines.size
  line = lines[i]

  # ── Strip standalone page headers / footers ───────────────────────────────
  if line.match?(/^Chapter \d+\..+•\s*\d+\s*$/)
    i += 1; next
  end

  if line.match?(/^report erratum\s*•\s*discuss\s*$/)
    i += 1; next
  end

  if line.match?(/^CHAPTER \d+\s*$/)
    i += 1; next
  end

  # ── Strip inline footer artifact (fused onto content line with no newline) ─
  line = line.gsub(/report erratum\s*•\s*discuss/, "").rstrip

  # ── Fix hyphenated line breaks ─────────────────────────────────────────────
  # Heuristic: line ends with hyphen AND next line starts with lowercase letter
  # AND next line is not itself a footer/header artifact.
  if line.end_with?("-") && i + 1 < lines.size
    next_line = lines[i + 1]
    is_artifact = next_line.match?(/^report erratum/) ||
                  next_line.match?(/^Chapter \d+\./) ||
                  next_line.match?(/^CHAPTER \d+/)
    if !is_artifact && next_line.match?(/^[a-z]/)
      line = line.delete_suffix("-") + next_line.lstrip
      i += 1
    end
  end

  cleaned << line
  i += 1
end

# ── Post-process ──────────────────────────────────────────────────────────────

text = cleaned.join("\n")

# Remove beta marker
text.gsub!("ß", "")

# Ensure blank line before/after headings.
# A heading is: non-blank, short (≤ 60 chars), no trailing sentence punctuation,
# not a code line (no leading spaces/$ or common code tokens), title-cased or short phrase.
lines2 = text.lines.map(&:chomp)
result = []

lines2.each_with_index do |line, idx|
  is_heading = !line.strip.empty? &&
               line.length <= 70 &&
               !line.match?(/[.!?,;]$/) &&
               !line.match?(/^\s/) &&           # not indented (not code)
               !line.match?(/^\$/) &&           # not shell prompt
               !line.match?(/^[a-z]/) &&        # starts with uppercase or special
               !line.match?(/^[#\[\{\(]/) &&    # not code
               line.match?(/[A-Z]/) &&           # has at least one capital
               line.strip.split.size <= 10      # short phrase, not a sentence

  if is_heading
    result << "" unless result.last == ""
    result << line
    result << ""
  else
    result << line
  end
end

text = result.join("\n")

# Collapse runs of 3+ blank lines to 2
text.gsub!(/\n{4,}/, "\n\n\n")

text = text.strip + "\n"

File.write(output, text)

original_size = File.size(input)
cleaned_size  = File.size(output)

# Report
remaining_footers = text.scan(/report erratum/).size
remaining_headers = text.lines.count { |l| l.match?(/^Chapter \d+\..+•/) }

puts "Done."
puts "  #{(original_size / 1024.0).round(1)} KB → #{(cleaned_size / 1024.0).round(1)} KB"
puts "  Remaining footers: #{remaining_footers}"
puts "  Remaining page headers: #{remaining_headers}"
puts "  Saved to #{output}"
