/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <rapids_logger/log_levels.h>

// Default to info level if not specified.
#if !defined(CUML_LOG_ACTIVE_LEVEL)
#define CUML_LOG_ACTIVE_LEVEL RAPIDS_LOGGER_LOG_LEVEL_INFO
#endif

// Macros for easier logging, similar to spdlog.
#define CUML_LOGGER_CALL(logger, level, ...) (logger).log(level, __VA_ARGS__)

#if CUML_LOG_ACTIVE_LEVEL <= RAPIDS_LOGGER_LOG_LEVEL_TRACE
#define CUML_LOG_TRACE(...) \
  CUML_LOGGER_CALL(ML::default_logger(), rapids_logger::level_enum::trace, __VA_ARGS__)
#else
#define CUML_LOG_TRACE(...) (void)0
#endif

#if CUML_LOG_ACTIVE_LEVEL <= RAPIDS_LOGGER_LOG_LEVEL_DEBUG
#define CUML_LOG_DEBUG(...) \
  CUML_LOGGER_CALL(ML::default_logger(), rapids_logger::level_enum::debug, __VA_ARGS__)
#else
#define CUML_LOG_DEBUG(...) (void)0
#endif

#if CUML_LOG_ACTIVE_LEVEL <= RAPIDS_LOGGER_LOG_LEVEL_INFO
#define CUML_LOG_INFO(...) CUML_LOGGER_CALL(ML::default_logger(), rapids_logger::level_enum::info, __VA_ARGS__)
#else
#define CUML_LOG_INFO(...) (void)0
#endif

#if CUML_LOG_ACTIVE_LEVEL <= RAPIDS_LOGGER_LOG_LEVEL_WARN
#define CUML_LOG_WARN(...) CUML_LOGGER_CALL(ML::default_logger(), rapids_logger::level_enum::warn, __VA_ARGS__)
#else
#define CUML_LOG_WARN(...) (void)0
#endif

#if CUML_LOG_ACTIVE_LEVEL <= RAPIDS_LOGGER_LOG_LEVEL_ERROR
#define CUML_LOG_ERROR(...) \
  CUML_LOGGER_CALL(ML::default_logger(), rapids_logger::level_enum::error, __VA_ARGS__)
#else
#define CUML_LOG_ERROR(...) (void)0
#endif

#if CUML_LOG_ACTIVE_LEVEL <= RAPIDS_LOGGER_LOG_LEVEL_CRITICAL
#define CUML_LOG_CRITICAL(...) \
  CUML_LOGGER_CALL(ML::default_logger(), rapids_logger::level_enum::critical, __VA_ARGS__)
#else
#define CUML_LOG_CRITICAL(...) (void)0
#endif
