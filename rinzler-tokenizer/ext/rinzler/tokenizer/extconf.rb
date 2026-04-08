require "mkmf"

$CFLAGS += " -std=c11 -Wall -O2"

create_makefile("rinzler/tokenizer/bpe_ext")
