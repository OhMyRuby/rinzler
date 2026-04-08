#!/usr/bin/env ruby
# frozen_string_literal: true
#
# clean_markdown.rb — convert Markdown files in corpus/ to plain text
#
# Strips:
#   - ATX headers:        ## Heading  →  Heading
#   - Setext underlines:  =======/------- lines (dropped)
#   - Horizontal rules:   ---, ***, ___ (dropped)
#   - Bold/italic:        **text**, *text*, __text__, _text_  →  text
#   - Inline code:        `code`  →  code
#   - Code fences:        ```lang ... ```  →  content preserved, fences dropped
#   - Table separators:   |---|---| rows (dropped)
#   - Table pipes:        | cell | cell |  →  cell  cell
#   - Links:              [text](url)  →  text
#   - Images:             ![alt](url)  →  alt
#   - Blockquotes:        > text  →  text
#   - HTML tags:          <tag>  →  (dropped)

def clean(text)
  lines    = text.lines
  output   = []
  in_fence = false

  lines.each do |line|
    l = line.chomp

    # Code fences — toggle state, drop the fence markers themselves
    if l.match?(/\A\s*`{3,}/)
      in_fence = !in_fence
      next
    end

    # Inside a code fence: preserve content as-is
    if in_fence
      output << l
      next
    end

    # Setext-style headings underline (===== or -----) — drop
    next if l.match?(/\A[=\-]{3,}\z/)

    # Horizontal rules (---, ***, ___) — drop
    next if l.match?(/\A\s*(\*\s*){3,}\z/) || l.match?(/\A\s*(-\s*){3,}\z/) || l.match?(/\A\s*(_\s*){3,}\z/)

    # Table separator rows (|---|---|) — drop
    next if l.match?(/\A\s*\|[\s\-:|]+\|\s*\z/)

    # ATX headers — strip leading # marks
    l = l.sub(/\A#+\s*/, "")

    # Blockquotes — strip leading >
    l = l.gsub(/(?:^|\s)>\s?/, " ").lstrip

    # Table rows — strip pipes, collapse to space-separated cells
    if l.match?(/\|/)
      l = l.gsub(/^\s*\|/, "").gsub(/\|\s*$/, "").gsub("|", "  ")
    end

    # Images: ![alt](url) → alt
    l = l.gsub(/!\[([^\]]*)\]\([^)]*\)/, '\1')

    # Links: [text](url) → text
    l = l.gsub(/\[([^\]]*)\]\([^)]*\)/, '\1')

    # Bold + italic: ***text*** or ___text___
    l = l.gsub(/\*{3}(.+?)\*{3}/, '\1')
    l = l.gsub(/_{3}(.+?)_{3}/, '\1')

    # Bold: **text** or __text__
    l = l.gsub(/\*{2}(.+?)\*{2}/, '\1')
    l = l.gsub(/_{2}(.+?)_{2}/, '\1')

    # Italic: *text* or _text_ (careful not to hit * in code)
    l = l.gsub(/\*(.+?)\*/, '\1')
    l = l.gsub(/(?<!\w)_(.+?)_(?!\w)/, '\1')

    # Inline code: `code`
    l = l.gsub(/`([^`]+)`/, '\1')

    # HTML tags
    l = l.gsub(/<[^>]+>/, "")

    # Trailing whitespace
    l = l.rstrip

    output << l
  end

  # Collapse runs of more than 2 blank lines to 2
  result = []
  blank_run = 0
  output.each do |l|
    if l.empty?
      blank_run += 1
      result << l if blank_run <= 2
    else
      blank_run = 0
      result << l
    end
  end

  result.join("\n").strip + "\n"
end

Dir.glob(File.join(__dir__, "*.md")).sort.each do |md_path|
  txt_path = md_path.sub(/\.md\z/, ".txt")
  original = File.read(md_path)
  cleaned  = clean(original)
  File.write(txt_path, cleaned)
  puts "  #{File.basename(md_path)} → #{File.basename(txt_path)} (#{original.size} → #{cleaned.size} chars)"
end
