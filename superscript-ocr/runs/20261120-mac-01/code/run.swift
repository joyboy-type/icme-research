import Foundation
import Vision
import ImageIO

let number = try NSRegularExpression(pattern: #"[0-9０-９]+(?:[,，\-–−][0-9０-９]+)*"#)
let encoder = JSONEncoder()
encoder.outputFormatting = [.sortedKeys]
struct Match: Codable {
    let text: String
    let x: Double
    let y: Double
}
struct Output: Codable {
    let path: String
    let revision: Int
    let matches: [Match]
}
for path in CommandLine.arguments.dropFirst() {
    let url = URL(fileURLWithPath: path)
    guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
          let image = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
        throw CocoaError(.fileReadCorruptFile)
    }
    let request = VNRecognizeTextRequest()
    request.revision = VNRecognizeTextRequestRevision3
    request.recognitionLevel = .accurate
    request.usesLanguageCorrection = false
    request.automaticallyDetectsLanguage = true
    try VNImageRequestHandler(cgImage: image, orientation: .up).perform([request])
    var matches: [Match] = []
    for observation in request.results ?? [] {
        guard let candidate = observation.topCandidates(1).first else { continue }
        let text = candidate.string
        for result in number.matches(in: text, range: NSRange(text.startIndex..., in: text)) {
            guard let range = Range(result.range, in: text),
                  let localized = try? candidate.boundingBox(for: range) else { continue }
            matches.append(Match(text: String(text[range]), x: localized.boundingBox.midX,
                                 y: 1 - localized.boundingBox.midY))
        }
    }
    print(String(decoding: try encoder.encode(Output(path: path, revision: request.revision, matches: matches)), as: UTF8.self))
}
