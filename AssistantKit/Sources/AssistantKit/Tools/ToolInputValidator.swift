import Foundation

/// Why a tool call's input was rejected.
public enum ToolInputError: Error, Equatable {
    /// The input wasn't valid JSON; holds the text as the model wrote it.
    case invalidJSON(String)
    /// The input parsed, but isn't a JSON object.
    case notAnObject
    /// A required field is absent.
    case missing(String)
    /// A field has the wrong JSON type; `expected` is the schema's type, e.g. "integer" or "string or null".
    case wrongType(field: String, expected: String)
    /// A field's value isn't one of the schema's `enum` values.
    case notAllowed(field: String, value: String)
    /// A field the schema doesn't define, where it allows no others.
    case unexpected(String)
}

/// Checks a tool call's input against the tool's JSON Schema before the tool runs.
///
/// It understands the subset of JSON Schema that tool definitions use: `type` (a name or a list of
/// names), `enum`, and, for objects, `properties`, `required` and `additionalProperties: false`;
/// arrays are checked item by item through `items`. Other keywords are ignored. Fields are checked
/// in a fixed order (missing required fields first, then the given fields by name), so the same
/// input always gives the same error.
///
/// Lenient mode is for small on-device models, which often get the JSON types slightly wrong. It
/// coerces a value whose meaning is unambiguous instead of rejecting it:
/// - `"5"` → 5 for an integer, `"2.5"` → 2.5 for a number, and 5 → `"5"` for a string;
/// - `"true"` / `"false"` → a boolean;
/// - a string that matches an `enum` value except for case or surrounding spaces → that value;
/// - a `null` field the schema doesn't allow to be null → left out (so a required one is missing);
/// - input given as a string that holds a JSON object → that object; `null` input → `{}`.
public enum ToolInputValidator {
    /// The call's input as a validated object (coerced in lenient mode), or why it was rejected.
    public static func validate(_ call: PendingToolCall, schema: JSONValue, lenient: Bool) -> Result<[String: JSONValue], ToolInputError> {
        guard var input = call.input else {
            return .failure(.invalidJSON(call.rawInput ?? ""))
        }
        if lenient {
            switch input {
            case .null:
                input = .object([:])
            case .string(let text):
                if let parsed = try? JSONValue.parse(text), parsed.objectValue != nil {
                    input = parsed
                }
            default:
                break
            }
        }
        guard case .object(let object) = input else {
            return .failure(.notAnObject)
        }
        do {
            return .success(try checkObject(object, schema: schema, path: "", lenient: lenient))
        } catch let error as ToolInputError {
            return .failure(error)
        } catch {
            return .failure(.notAnObject)
        }
    }

    /// The tool result content for a rejected input: `{"INVALID_JSON": raw}` for input that didn't
    /// parse (the form Claude's tool-use docs recommend), otherwise `{"error": "..."}` with a short
    /// explanation the model can act on.
    public static func errorContent(_ error: ToolInputError) -> JSONValue {
        switch error {
        case .invalidJSON(let raw):
            return ["INVALID_JSON": .string(raw)]
        default:
            return ["error": .string(message(for: error))]
        }
    }

    /// A one-sentence explanation of `error`.
    static func message(for error: ToolInputError) -> String {
        switch error {
        case .invalidJSON:
            return "The input is not valid JSON."
        case .notAnObject:
            return "The input must be a JSON object."
        case .missing(let field):
            return "Missing required field \"\(field)\"."
        case .wrongType(let field, let expected):
            return "Field \"\(field)\" must be \(describe(expected))."
        case .notAllowed(let field, let value):
            return "Field \"\(field)\" can't be \"\(value)\"; use one of the values the tool's schema lists."
        case .unexpected(let field):
            return "Unexpected field \"\(field)\"; use only the fields the tool's schema lists."
        }
    }

    // MARK: - Checking

    private static func checkObject(_ object: [String: JSONValue], schema: JSONValue, path: String, lenient: Bool) throws -> [String: JSONValue] {
        let properties = schema["properties"]?.objectValue ?? [:]
        let required = (schema["required"]?.arrayValue ?? []).compactMap(\.stringValue)
        let closed = schema["additionalProperties"]?.boolValue == false

        var fields = object
        if lenient {
            // Small models write `"due": null` for an optional field they mean to leave out.
            for (key, value) in object where value.isNull {
                let allowsNull = properties[key].map { typeNames(of: $0)?.contains("null") ?? true } ?? false
                if !allowsNull { fields.removeValue(forKey: key) }
            }
        }

        for name in required where fields[name] == nil {
            throw ToolInputError.missing(join(path, name))
        }

        var checked: [String: JSONValue] = [:]
        for key in fields.keys.sorted() {
            let value = fields[key] ?? .null
            let field = join(path, key)
            if let property = properties[key] {
                checked[key] = try check(value, schema: property, path: field, lenient: lenient)
            } else if closed {
                throw ToolInputError.unexpected(field)
            } else if let extra = schema["additionalProperties"], extra.objectValue != nil {
                checked[key] = try check(value, schema: extra, path: field, lenient: lenient)
            } else {
                checked[key] = value
            }
        }
        return checked
    }

    private static func check(_ value: JSONValue, schema: JSONValue, path: String, lenient: Bool) throws -> JSONValue {
        var result = value
        if let types = typeNames(of: schema) {
            guard let matched = types.lazy.compactMap({ match(value, type: $0, lenient: lenient) }).first else {
                throw ToolInputError.wrongType(field: path, expected: types.joined(separator: " or "))
            }
            result = matched
        }

        if let allowed = schema["enum"]?.arrayValue {
            if !allowed.contains(result) {
                guard lenient, let text = result.stringValue, let close = closeEnumValue(text, allowed: allowed) else {
                    throw ToolInputError.notAllowed(field: path, value: render(result))
                }
                result = close
            }
        }

        switch result {
        case .object(let object) where typeNames(of: schema)?.contains("object") == true || schema["properties"] != nil:
            return .object(try checkObject(object, schema: schema, path: path, lenient: lenient))
        case .array(let items):
            guard let itemSchema = schema["items"], itemSchema.objectValue != nil else { return result }
            var checked: [JSONValue] = []
            for (index, item) in items.enumerated() {
                checked.append(try check(item, schema: itemSchema, path: "\(path)[\(index)]", lenient: lenient))
            }
            return .array(checked)
        default:
            return result
        }
    }

    /// `value` as an instance of the JSON Schema `type`, coerced in lenient mode, or nil if it isn't one.
    private static func match(_ value: JSONValue, type: String, lenient: Bool) -> JSONValue? {
        switch type {
        case "string":
            switch value {
            case .string:
                return value
            case .int(let number) where lenient:
                return .string(String(number))
            case .double(let number) where lenient && number.isFinite:
                if let whole = value.intValue { return .string(String(whole)) }
                return .string(String(number))
            default:
                return nil
            }
        case "integer":
            switch value {
            case .int:
                return value
            case .double:
                // JSON Schema counts 5.0 as an integer.
                return value.intValue.map { JSONValue.int($0) }
            case .string(let text) where lenient:
                return integer(from: text).map { JSONValue.int($0) }
            default:
                return nil
            }
        case "number":
            switch value {
            case .int, .double:
                return value
            case .string(let text) where lenient:
                let trimmed = text.trimmed
                if let whole = Int(trimmed) { return .int(whole) }
                if let number = Double(trimmed), number.isFinite { return .double(number) }
                return nil
            default:
                return nil
            }
        case "boolean":
            switch value {
            case .bool:
                return value
            case .string(let text) where lenient:
                switch text.trimmed.lowercased() {
                case "true": return .bool(true)
                case "false": return .bool(false)
                default: return nil
                }
            default:
                return nil
            }
        case "object":
            return value.objectValue != nil ? value : nil
        case "array":
            return value.arrayValue != nil ? value : nil
        case "null":
            return value.isNull ? value : nil
        default:
            // A type this validator doesn't know: don't reject what it can't judge.
            return value
        }
    }

    private static func integer(from text: String) -> Int? {
        let trimmed = text.trimmed
        if let whole = Int(trimmed) { return whole }
        if let number = Double(trimmed), number.isFinite, number.rounded() == number, abs(number) < 9e15 {
            return Int(number)
        }
        return nil
    }

    /// The single `enum` string that equals `text` ignoring case and surrounding whitespace.
    private static func closeEnumValue(_ text: String, allowed: [JSONValue]) -> JSONValue? {
        let wanted = text.trimmed.lowercased()
        let matches = allowed.filter { $0.stringValue?.lowercased() == wanted }
        return matches.count == 1 ? matches[0] : nil
    }

    /// The schema's `type` as a list of names, or nil when it names none.
    private static func typeNames(of schema: JSONValue) -> [String]? {
        switch schema["type"] {
        case .string(let name)?:
            return [name]
        case .array(let names)?:
            let list = names.compactMap(\.stringValue)
            return list.isEmpty ? nil : list
        default:
            return nil
        }
    }

    private static func join(_ path: String, _ key: String) -> String {
        path.isEmpty ? key : path + "." + key
    }

    private static func render(_ value: JSONValue) -> String {
        if let text = value.stringValue { return text }
        guard let data = try? value.serialized() else { return "" }
        return String(decoding: data, as: UTF8.self)
    }

    /// "integer" → "an integer", "string or null" → "a string or null".
    private static func describe(_ expected: String) -> String {
        expected
            .components(separatedBy: " or ")
            .map { name -> String in
                switch name {
                case "null": return "null"
                case "integer", "object", "array": return "an \(name)"
                default: return "a \(name)"
                }
            }
            .joined(separator: " or ")
    }
}
