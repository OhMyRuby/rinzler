# frozen_string_literal: true

require "logger"

module Rinzler
  # Rinzler.logger — shared logger for all Rinzler gems.
  #
  # Defaults to $stdout at INFO level with a compact format.
  # Configure before loading other gems to control output:
  #
  #   Rinzler.logger.level = Logger::DEBUG
  #   Rinzler.logger = Logger.new("rinzler.log")
  #
  class << self
    attr_writer :logger

    def logger
      @logger ||= begin
        log = Logger.new($stdout)
        log.level     = Logger::INFO
        log.progname  = "rinzler"
        log.formatter = proc do |severity, time, progname, msg|
          ts = time.strftime("%H:%M:%S")
          "[#{ts}] #{severity.ljust(5)} -- #{progname}: #{msg}\n"
        end
        log
      end
    end
  end
end
