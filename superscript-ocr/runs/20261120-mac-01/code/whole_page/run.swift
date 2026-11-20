import Foundation
import Vision
import ImageIO

struct Line: Codable {
    let text: String
    let confidence: Float
    let box: [Double]
}

struct Page: Codable {
    let pageID: String
    let pixelWidth: Int
    let pixelHeight: Int
    let revision: Int
    let elapsedMS: Double
    let lines: [Line]
}

let encoder = JSONEncoder()
encoder.outputFormatting = [.sortedKeys]
let images = CommandLine.arguments.dropFirst()
for path in images {
    let url = URL(fileURLWithPath: path)
    guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
          let image = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
        throw CocoaError(.fileReadCorruptFile)
    }
    let request = VNRecognizeTextRequest()
    request.recognitionLevel = .accurate
    request.usesLanguageCorrection = false
    request.automaticallyDetectsLanguage = true
    let started = ProcessInfo.processInfo.systemUptime
    try VNImageRequestHandler(cgImage: image, orientation: .up).perform([request])
    let elapsedMS = (ProcessInfo.processInfo.systemUptime - started) * 1000
    let lines: [Line] = (request.results ?? []).compactMap { observation in
        guard let candidate = observation.topCandidates(1).first else { return nil }
        let b = observation.boundingBox
        return Line(text: candidate.string, confidence: candidate.confidence,
                    box: [b.minX, 1 - b.maxY, b.width, b.height])
    }
    let page = Page(pageID: url.deletingPathExtension().lastPathComponent,
                    pixelWidth: image.width, pixelHeight: image.height,
                    revision: request.revision, elapsedMS: elapsedMS, lines: lines)
    print(String(decoding: try encoder.encode(page), as: UTF8.self))
}
