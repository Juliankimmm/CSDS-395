import UIKit

actor ImageCache {
    static let shared = ImageCache()

    private var store: [String: UIImage] = [:]

    func image(for id: String) -> UIImage? {
        return store[id]
    }

    func setImage(_ image: UIImage, for id: String) {
        store[id] = image
    }
}
